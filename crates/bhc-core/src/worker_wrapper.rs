//! Worker/wrapper transformation for Core IR.
//!
//! For each function with strict arguments (as determined by demand analysis),
//! wraps the body in `case` expressions that force those arguments eagerly.
//! This avoids unnecessary thunk allocation in the Default profile.
//!
//! ## Transformation
//!
//! ```text
//! -- Before: f = \x y z -> body   (sig: [Strict, Lazy, Strict])
//! -- After:  f = \x y z -> case x of x' -> case z of z' -> body[x→x', z→z']
//! ```
//!
//! The `case` forces the scrutinee to WHNF. Unlike GHC, BHC does not
//! distinguish boxed `Int` from unboxed `Int#` at the LLVM level
//! (everything is i64), so we don't create separate worker functions
//! with unboxed signatures. Instead, the case-wrapping tells codegen
//! to evaluate the argument immediately.
//!
//! ## References
//!
//! See H26-SPEC Section 6 (evaluation strategy) and BHC-RULE-013
//! (optimization guidelines).

use rustc_hash::FxHashMap;

use bhc_intern::Symbol;
use bhc_span::Span;

use crate::simplify::expr_util::fresh_var_id;
use crate::simplify::subst::substitute;
use crate::{Alt, AltCon, Bind, CoreModule, Expr, Var, VarId};
use crate::demand::{Demand, DemandResult};

/// Apply worker/wrapper transformation to a Core module.
///
/// For each function with at least one strict argument, wraps the body
/// in case expressions that force the strict arguments.
///
/// Returns the number of functions transformed.
pub fn apply_worker_wrapper(module: &mut CoreModule, demands: &DemandResult) -> usize {
    let mut count = 0;

    for bind in &mut module.bindings {
        match bind {
            Bind::NonRec(var, rhs) => {
                if should_skip(var) {
                    continue;
                }
                if let Some(sig) = demands.get(&var.id) {
                    if sig.args.contains(&Demand::Strict) && transform_function(rhs, sig) {
                        count += 1;
                    }
                }
            }
            Bind::Rec(pairs) => {
                for (var, rhs) in pairs.iter_mut() {
                    if should_skip(var) {
                        continue;
                    }
                    if let Some(sig) = demands.get(&var.id) {
                        if sig.args.contains(&Demand::Strict) && transform_function(rhs, sig) {
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    count
}

/// Check if a binding should be skipped (protected names).
fn should_skip(var: &Var) -> bool {
    let name = var.name.as_str();
    name == "main"
        || name.starts_with('$')
        || name.starts_with("bhc_")
        || name.contains("::")
}

/// Transform a single function by wrapping strict args in case expressions.
///
/// Returns true if the transformation was applied.
fn transform_function(
    expr: &mut Box<Expr>,
    sig: &crate::demand::DemandSig,
) -> bool {
    // Peel off lambda parameters
    let mut params: Vec<(Var, Span)> = Vec::new();
    let mut current = expr.as_mut();

    loop {
        match current {
            Expr::Lam(x, body, span) => {
                params.push((x.clone(), *span));
                current = body.as_mut();
            }
            Expr::TyLam(_, body, _) => {
                // Skip type lambdas
                current = body.as_mut();
            }
            _ => break,
        }
    }

    if params.is_empty() {
        return false;
    }

    // Collect strict parameters (those with Demand::Strict in the signature)
    let strict_params: Vec<&Var> = params
        .iter()
        .zip(sig.args.iter())
        .filter_map(|((var, _), demand)| {
            if *demand == Demand::Strict {
                Some(var)
            } else {
                None
            }
        })
        .collect();

    if strict_params.is_empty() {
        return false;
    }

    // Build case-wrappers for each strict parameter.
    // We wrap from innermost to outermost so the first strict param
    // is evaluated first.
    //
    // For each strict param x, we generate:
    //   case x of x' -> body[x→x']
    //
    // The fresh variable x' ensures that after the case, uses of x
    // in the body are bound to the evaluated value.

    let mut subst_map: FxHashMap<VarId, Expr> = FxHashMap::default();
    let mut case_wrappers: Vec<(Var, Var)> = Vec::new(); // (original, fresh)

    for param in &strict_params {
        let fresh_id = fresh_var_id();
        let fresh_name = Symbol::intern(&format!("{}$ww", param.name.as_str()));
        let fresh_var = Var::new(fresh_name, fresh_id, param.ty.clone());

        subst_map.insert(
            param.id,
            Expr::Var(fresh_var.clone(), Span::default()),
        );
        case_wrappers.push(((*param).clone(), fresh_var));
    }

    // Extract the body (innermost non-lambda expression)
    let body = take_body(expr.as_mut());

    // Apply substitution to body
    let substituted_body = substitute(body, &subst_map);

    // Build nested case expressions from inside out
    let mut wrapped = substituted_body;
    for (orig, fresh) in case_wrappers.into_iter().rev() {
        let result_ty = wrapped.ty();
        let scrutinee = Expr::Var(orig, Span::default());
        wrapped = Expr::Case(
            Box::new(scrutinee),
            vec![Alt {
                con: AltCon::Default,
                binders: vec![fresh],
                rhs: wrapped,
            }],
            result_ty,
            Span::default(),
        );
    }

    // Reconstruct the lambda chain with the wrapped body
    put_body(expr.as_mut(), wrapped);

    true
}

/// Extract the body from a nested lambda/type-lambda expression.
fn take_body(expr: &mut Expr) -> Expr {
    match expr {
        Expr::Lam(_, body, _) | Expr::TyLam(_, body, _) => take_body(body.as_mut()),
        other => {
            // Replace with a dummy and return the original
            std::mem::replace(
                other,
                Expr::Lit(crate::Literal::Int(0), crate::Ty::Error, Span::default()),
            )
        }
    }
}

/// Put a new body into a nested lambda/type-lambda expression.
fn put_body(expr: &mut Expr, new_body: Expr) {
    match expr {
        Expr::Lam(_, body, _) | Expr::TyLam(_, body, _) => put_body(body.as_mut(), new_body),
        other => {
            *other = new_body;
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use crate::demand::DemandSig;
    use crate::{CoreModule, Literal};
    use bhc_types::Ty;

    fn mk_var(name: &str, id: u32) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id as usize), Ty::Error)
    }

    fn mk_var_expr(name: &str, id: u32) -> Expr {
        Expr::Var(mk_var(name, id), Span::default())
    }

    fn mk_int(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
    }

    fn mk_lam(name: &str, id: u32, body: Expr) -> Expr {
        Expr::Lam(mk_var(name, id), Box::new(body), Span::default())
    }

    fn mk_binop(op: &str, a: Expr, b: Expr) -> Expr {
        Expr::App(
            Box::new(Expr::App(
                Box::new(mk_var_expr(op, 0)),
                Box::new(a),
                Span::default(),
            )),
            Box::new(b),
            Span::default(),
        )
    }

    fn mk_module(bindings: Vec<Bind>) -> CoreModule {
        CoreModule {
            name: Symbol::intern("Test"),
            bindings,
            exports: vec![],
            overloaded_strings: false,
            constructors: vec![],
        }
    }

    // --------------------------------------------------------
    // Test 1: One strict arg → single case wrapper
    // --------------------------------------------------------
    #[test]
    fn test_single_strict_arg() {
        // f = \x -> x + 1
        let body = mk_binop("+", mk_var_expr("x", 1), mk_int(1));
        let func = mk_lam("x", 1, body);
        let mut module = mk_module(vec![Bind::NonRec(mk_var("f", 100), Box::new(func))]);

        let mut demands = DemandResult::default();
        demands.insert(
            VarId::new(100),
            DemandSig {
                args: vec![Demand::Strict],
            },
        );

        let count = apply_worker_wrapper(&mut module, &demands);
        assert_eq!(count, 1);

        // Verify the transformation: f = \x -> case x of x' -> x' + 1
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            if let Expr::Lam(_, inner, _) = rhs.as_ref() {
                assert!(
                    matches!(inner.as_ref(), Expr::Case(_, _, _, _)),
                    "expected Case wrapper, got {:?}",
                    inner
                );
            } else {
                panic!("expected Lam");
            }
        } else {
            panic!("expected NonRec");
        }
    }

    // --------------------------------------------------------
    // Test 2: Multiple strict args → nested case wrappers
    // --------------------------------------------------------
    #[test]
    fn test_multiple_strict_args() {
        // f = \x -> \y -> x + y
        let body = mk_binop("+", mk_var_expr("x", 1), mk_var_expr("y", 2));
        let func = mk_lam("x", 1, mk_lam("y", 2, body));
        let mut module = mk_module(vec![Bind::NonRec(mk_var("f", 100), Box::new(func))]);

        let mut demands = DemandResult::default();
        demands.insert(
            VarId::new(100),
            DemandSig {
                args: vec![Demand::Strict, Demand::Strict],
            },
        );

        let count = apply_worker_wrapper(&mut module, &demands);
        assert_eq!(count, 1);

        // Verify nested case wrappers
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            if let Expr::Lam(_, inner, _) = rhs.as_ref() {
                if let Expr::Lam(_, inner2, _) = inner.as_ref() {
                    // Should be: case x of x' -> case y of y' -> x' + y'
                    assert!(
                        matches!(inner2.as_ref(), Expr::Case(_, _, _, _)),
                        "expected outer Case, got {:?}",
                        inner2
                    );
                } else {
                    panic!("expected inner Lam");
                }
            } else {
                panic!("expected Lam");
            }
        }
    }

    // --------------------------------------------------------
    // Test 3: No strict args → unchanged
    // --------------------------------------------------------
    #[test]
    fn test_no_strict_args() {
        let func = mk_lam("x", 1, mk_var_expr("x", 1));
        let mut module = mk_module(vec![Bind::NonRec(mk_var("f", 100), Box::new(func))]);

        let mut demands = DemandResult::default();
        demands.insert(
            VarId::new(100),
            DemandSig {
                args: vec![Demand::Lazy],
            },
        );

        let count = apply_worker_wrapper(&mut module, &demands);
        assert_eq!(count, 0);
    }

    // --------------------------------------------------------
    // Test 4: Protected names → skipped
    // --------------------------------------------------------
    #[test]
    fn test_protected_names_skipped() {
        let func = mk_lam("x", 1, mk_binop("+", mk_var_expr("x", 1), mk_int(1)));
        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 100), Box::new(func))]);

        let mut demands = DemandResult::default();
        demands.insert(
            VarId::new(100),
            DemandSig {
                args: vec![Demand::Strict],
            },
        );

        let count = apply_worker_wrapper(&mut module, &demands);
        assert_eq!(count, 0); // main is protected
    }

    // --------------------------------------------------------
    // Test 5: Non-function (no lambdas) → skipped
    // --------------------------------------------------------
    #[test]
    fn test_non_function_skipped() {
        let mut module = mk_module(vec![Bind::NonRec(mk_var("x", 100), Box::new(mk_int(42)))]);

        // Even if we have a demand sig for it, no transformation
        let demands = DemandResult::default();
        let count = apply_worker_wrapper(&mut module, &demands);
        assert_eq!(count, 0);
    }

    // --------------------------------------------------------
    // Test 6: Mixed strict/lazy → only strict args get case wrappers
    // --------------------------------------------------------
    #[test]
    fn test_mixed_strict_lazy() {
        // f = \x -> \y -> \z -> x + z  (y is lazy)
        let body = mk_binop("+", mk_var_expr("x", 1), mk_var_expr("z", 3));
        let func = mk_lam("x", 1, mk_lam("y", 2, mk_lam("z", 3, body)));
        let mut module = mk_module(vec![Bind::NonRec(mk_var("f", 100), Box::new(func))]);

        let mut demands = DemandResult::default();
        demands.insert(
            VarId::new(100),
            DemandSig {
                args: vec![Demand::Strict, Demand::Lazy, Demand::Strict],
            },
        );

        let count = apply_worker_wrapper(&mut module, &demands);
        assert_eq!(count, 1);

        // The innermost body should have case wrappers for x and z but not y
        // f = \x -> \y -> \z -> case x of x' -> case z of z' -> x' + z'
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            // Navigate through lambdas
            let mut expr = rhs.as_ref();
            for _ in 0..3 {
                if let Expr::Lam(_, body, _) = expr {
                    expr = body.as_ref();
                } else {
                    panic!("expected Lam");
                }
            }
            // Should be a Case (outer wrapper for x)
            assert!(
                matches!(expr, Expr::Case(_, _, _, _)),
                "expected Case, got {:?}",
                expr
            );
        }
    }

    // --------------------------------------------------------
    // Test 7: Recursive binding with strict args
    // --------------------------------------------------------
    #[test]
    fn test_recursive_binding() {
        // go = \n -> \acc -> n + acc
        let body = mk_binop("+", mk_var_expr("n", 1), mk_var_expr("acc", 2));
        let func = mk_lam("n", 1, mk_lam("acc", 2, body));
        let mut module = mk_module(vec![Bind::Rec(vec![(mk_var("go", 100), Box::new(func))])]);

        let mut demands = DemandResult::default();
        demands.insert(
            VarId::new(100),
            DemandSig {
                args: vec![Demand::Strict, Demand::Strict],
            },
        );

        let count = apply_worker_wrapper(&mut module, &demands);
        assert_eq!(count, 1);
    }

    // --------------------------------------------------------
    // Test 8: $-prefixed names are skipped
    // --------------------------------------------------------
    #[test]
    fn test_dollar_prefix_skipped() {
        let func = mk_lam("x", 1, mk_binop("+", mk_var_expr("x", 1), mk_int(1)));
        let mut module = mk_module(vec![Bind::NonRec(
            mk_var("$derived_show_Foo", 100),
            Box::new(func),
        )]);

        let mut demands = DemandResult::default();
        demands.insert(
            VarId::new(100),
            DemandSig {
                args: vec![Demand::Strict],
            },
        );

        let count = apply_worker_wrapper(&mut module, &demands);
        assert_eq!(count, 0);
    }
}
