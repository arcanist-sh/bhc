//! Dictionary specialization for Core IR.
//!
//! When BHC compiles typeclass-polymorphic code, it generates dictionary-passing
//! code where method selections are `$sel_N` applied to dictionary tuples. When
//! the dictionary is a known tuple constructor application, we can inline the
//! method selection directly, eliminating both the tuple allocation and the
//! selection overhead.
//!
//! ## Example
//!
//! ```text
//! -- Before
//! let $dEq_Int = App(App(Var("(,)"), eq_impl), ne_impl)
//! in ... $sel_0 $dEq_Int ...
//!
//! -- After
//! let $dEq_Int = App(App(Var("(,)"), eq_impl), ne_impl)
//! in ... eq_impl ...
//! ```
//!
//! The dead `$dEq_Int` binding is cleaned up by the subsequent simplifier pass.
//!
//! ## References
//!
//! See BHC-RULE-013 (optimization guidelines) and H26-SPEC Section 5.

use rustc_hash::FxHashMap;

use crate::{Bind, CoreModule, Expr, VarId};

/// Environment mapping dictionary `VarId`s to their extracted tuple fields.
type DictEnv = FxHashMap<VarId, Vec<Expr>>;

/// Apply dictionary specialization to a Core module.
///
/// Walks all bindings, collecting known dictionary bindings (variables starting
/// with `$d` whose RHS is a tuple constructor application). For each
/// `App($sel_N, $d)` where `$d` maps to a known tuple, replaces the entire
/// application with the Nth field of the tuple.
///
/// Returns the number of specializations performed.
pub fn specialize_dictionaries(module: &mut CoreModule) -> usize {
    let mut count = 0;

    for bind in &mut module.bindings {
        match bind {
            Bind::NonRec(_, rhs) => {
                let mut env = DictEnv::default();
                count += specialize_expr(rhs.as_mut(), &mut env);
            }
            Bind::Rec(pairs) => {
                let mut env = DictEnv::default();
                for (_, rhs) in pairs.iter_mut() {
                    count += specialize_expr(rhs.as_mut(), &mut env);
                }
            }
        }
    }

    count
}

/// Recursively specialize dictionary method selections in an expression.
///
/// Maintains a growing `DictEnv` as it descends into let-bindings.
fn specialize_expr(expr: &mut Expr, env: &mut DictEnv) -> usize {
    match expr {
        // Let bindings: check if this binds a known dictionary
        Expr::Let(bind, body, _) => {
            let mut count = 0;

            match bind.as_mut() {
                Bind::NonRec(var, rhs) => {
                    // First, recurse into the RHS itself
                    count += specialize_expr(rhs.as_mut(), env);

                    // Check if this is a dictionary binding ($d prefix + tuple RHS)
                    let name = var.name.as_str();
                    if name.starts_with("$d") {
                        if let Some(fields) = extract_tuple_fields(rhs) {
                            env.insert(var.id, fields);
                        }
                    }
                }
                Bind::Rec(pairs) => {
                    for (_, rhs) in pairs.iter_mut() {
                        count += specialize_expr(rhs.as_mut(), env);
                    }
                }
            }

            // Recurse into the body (where $sel_N calls typically live)
            count += specialize_expr(body.as_mut(), env);
            count
        }

        // Application: check for $sel_N applied to a known dictionary
        Expr::App(..) => {
            // First try to resolve this as a $sel_N application
            if let Expr::App(func, arg, _) = expr {
                if let Some(replacement) = try_inline_selection(func, arg, env) {
                    *expr = replacement;
                    return 1 + specialize_expr(expr, env);
                }
            }

            // Otherwise, recurse into sub-expressions
            let mut count = 0;
            if let Expr::App(func, arg, _) = expr {
                count += specialize_expr(func.as_mut(), env);
                count += specialize_expr(arg.as_mut(), env);
            }

            // After recursing, try again in case recursion exposed new opportunities
            if let Expr::App(func, arg, _) = expr {
                if let Some(replacement) = try_inline_selection(func, arg, env) {
                    *expr = replacement;
                    count += 1;
                }
            }

            count
        }

        // Recurse structurally through all other forms
        Expr::Lam(_, body, _) | Expr::TyLam(_, body, _) | Expr::Lazy(body, _) => {
            specialize_expr(body.as_mut(), env)
        }
        Expr::TyApp(inner, _, _) | Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) => {
            specialize_expr(inner.as_mut(), env)
        }
        Expr::Case(scrut, alts, _, _) => {
            let mut count = specialize_expr(scrut.as_mut(), env);
            for alt in alts.iter_mut() {
                count += specialize_expr(&mut alt.rhs, env);
            }
            count
        }
        Expr::Var(_, _) | Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => 0,
    }
}

/// Try to inline a `$sel_N dict` application.
///
/// Returns `Some(field_expr)` if `func` is a `$sel_N` variable and `arg`
/// resolves to a known dictionary (either a variable in `env` or a direct
/// tuple constructor application).
fn try_inline_selection(func: &Expr, arg: &Expr, env: &DictEnv) -> Option<Expr> {
    // Check that func is a $sel_N variable
    let sel_index = match func {
        Expr::Var(v, _) => parse_sel_index(v.name.as_str())?,
        _ => return None,
    };

    // Try to resolve the argument to tuple fields
    if let Expr::Var(v, _) = arg {
        // Case 1: arg is a variable that maps to a known dictionary
        let fields = env.get(&v.id)?;
        fields.get(sel_index).cloned()
    } else {
        // Case 2: arg is directly a tuple constructor application
        let direct = extract_tuple_fields(arg)?;
        direct.into_iter().nth(sel_index)
    }
}

/// Parse a `$sel_N` name and return the field index N.
///
/// Returns `None` if the name doesn't match the `$sel_` pattern.
fn parse_sel_index(name: &str) -> Option<usize> {
    let suffix = name.strip_prefix("$sel_")?;
    suffix.parse::<usize>().ok()
}

/// Extract tuple fields from a tuple constructor application.
///
/// Recognizes expressions of the form:
/// `App(App(Var("(,)"), field0), field1)` for pairs,
/// `App(App(App(Var("(,,)"), f0), f1), f2)` for triples, etc.
///
/// Returns `None` if the expression is not a tuple constructor application.
fn extract_tuple_fields(expr: &Expr) -> Option<Vec<Expr>> {
    let mut args = Vec::new();
    let mut current = expr;

    // Peel App nodes from the outside in
    while let Expr::App(func, arg, _) = current {
        args.push(arg.as_ref().clone());
        current = func.as_ref();
    }

    // Check that the head is a tuple constructor
    if let Expr::Var(v, _) = current {
        if is_tuple_constructor(v.name.as_str()) {
            // Args were collected outermost-first, reverse to get field order
            args.reverse();
            return Some(args);
        }
    }

    None
}

/// Check if a name is a tuple constructor: `(,)`, `(,,)`, `(,,,)`, etc.
fn is_tuple_constructor(name: &str) -> bool {
    if !name.starts_with("(,") || !name.ends_with(')') {
        return false;
    }
    // Everything between ( and ) should be commas
    let inner = &name[1..name.len() - 1];
    !inner.is_empty() && inner.chars().all(|c| c == ',')
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;
    use crate::{Alt, AltCon, CoreModule, Literal, Var, VarId};

    fn mk_var(name: &str, id: u32) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id as usize), Ty::Error)
    }

    fn mk_var_expr(name: &str, id: u32) -> Expr {
        Expr::Var(mk_var(name, id), Span::default())
    }

    fn mk_int(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
    }

    fn mk_app(f: Expr, a: Expr) -> Expr {
        Expr::App(Box::new(f), Box::new(a), Span::default())
    }

    fn mk_let_nonrec(name: &str, id: u32, rhs: Expr, body: Expr) -> Expr {
        Expr::Let(
            Box::new(Bind::NonRec(mk_var(name, id), Box::new(rhs))),
            Box::new(body),
            Span::default(),
        )
    }

    fn mk_lam(name: &str, id: u32, body: Expr) -> Expr {
        Expr::Lam(mk_var(name, id), Box::new(body), Span::default())
    }

    /// Build a tuple constructor application: (,) a b → App(App(Var("(,)"), a), b)
    fn mk_tuple2(a: Expr, b: Expr) -> Expr {
        mk_app(mk_app(mk_var_expr("(,)", 0), a), b)
    }

    /// Build a 3-tuple: (,,) a b c
    fn mk_tuple3(a: Expr, b: Expr, c: Expr) -> Expr {
        mk_app(mk_app(mk_app(mk_var_expr("(,,)", 0), a), b), c)
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

    // Helper to check if expression is a specific variable
    fn is_var_with_id(expr: &Expr, expected_id: u32) -> bool {
        matches!(expr, Expr::Var(v, _) if v.id == VarId::new(expected_id as usize))
    }

    // Helper to check if expression is a specific int literal
    fn is_int_lit(expr: &Expr, expected: i64) -> bool {
        matches!(expr, Expr::Lit(Literal::Int(n), _, _) if *n == expected)
    }

    // --------------------------------------------------------
    // Test 1: is_tuple_constructor
    // --------------------------------------------------------
    #[test]
    fn test_is_tuple_constructor() {
        assert!(is_tuple_constructor("(,)"));
        assert!(is_tuple_constructor("(,,)"));
        assert!(is_tuple_constructor("(,,,)"));
        assert!(!is_tuple_constructor("()"));
        assert!(!is_tuple_constructor("(+)"));
        assert!(!is_tuple_constructor("Just"));
        assert!(!is_tuple_constructor("(,x)"));
        assert!(!is_tuple_constructor(""));
    }

    // --------------------------------------------------------
    // Test 2: parse_sel_index
    // --------------------------------------------------------
    #[test]
    fn test_parse_sel_index() {
        assert_eq!(parse_sel_index("$sel_0"), Some(0));
        assert_eq!(parse_sel_index("$sel_1"), Some(1));
        assert_eq!(parse_sel_index("$sel_12"), Some(12));
        assert_eq!(parse_sel_index("$sel_"), None);
        assert_eq!(parse_sel_index("$sel_abc"), None);
        assert_eq!(parse_sel_index("other"), None);
        assert_eq!(parse_sel_index("$derived_show"), None);
    }

    // --------------------------------------------------------
    // Test 3: extract_tuple_fields for pairs
    // --------------------------------------------------------
    #[test]
    fn test_extract_tuple_fields_pair() {
        let pair = mk_tuple2(mk_int(1), mk_int(2));
        let fields = extract_tuple_fields(&pair).unwrap();
        assert_eq!(fields.len(), 2);
        assert!(is_int_lit(&fields[0], 1));
        assert!(is_int_lit(&fields[1], 2));
    }

    // --------------------------------------------------------
    // Test 4: extract_tuple_fields for triples
    // --------------------------------------------------------
    #[test]
    fn test_extract_tuple_fields_triple() {
        let triple = mk_tuple3(mk_int(10), mk_int(20), mk_int(30));
        let fields = extract_tuple_fields(&triple).unwrap();
        assert_eq!(fields.len(), 3);
        assert!(is_int_lit(&fields[0], 10));
        assert!(is_int_lit(&fields[1], 20));
        assert!(is_int_lit(&fields[2], 30));
    }

    // --------------------------------------------------------
    // Test 5: extract_tuple_fields returns None for non-tuples
    // --------------------------------------------------------
    #[test]
    fn test_extract_tuple_fields_non_tuple() {
        assert!(extract_tuple_fields(&mk_int(42)).is_none());
        assert!(extract_tuple_fields(&mk_var_expr("x", 1)).is_none());
        // App with non-tuple head
        let app = mk_app(mk_var_expr("Just", 0), mk_int(1));
        assert!(extract_tuple_fields(&app).is_none());
    }

    // --------------------------------------------------------
    // Test 6: Single method class — $sel_0 on known 2-element dict
    // --------------------------------------------------------
    #[test]
    fn test_single_method_selection() {
        // let $dEq_Int = (,) eq_impl ne_impl
        // in $sel_0 $dEq_Int
        let dict_rhs = mk_tuple2(mk_var_expr("eq_impl", 10), mk_var_expr("ne_impl", 11));
        let sel_call = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("$dEq_Int", 50));
        let expr = mk_let_nonrec("$dEq_Int", 50, dict_rhs, sel_call);

        let mut module = mk_module(vec![Bind::NonRec(mk_var("test", 100), Box::new(expr))]);
        let count = specialize_dictionaries(&mut module);

        assert_eq!(count, 1);

        // After specialization, the body should be eq_impl (VarId 10)
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            if let Expr::Let(_, body, _) = rhs.as_ref() {
                assert!(
                    is_var_with_id(body, 10),
                    "expected eq_impl (id=10), got {:?}",
                    body
                );
            } else {
                panic!("expected Let");
            }
        } else {
            panic!("expected NonRec");
        }
    }

    // --------------------------------------------------------
    // Test 7: Multi-method class — $sel_0 and $sel_1 on 3-element dict
    // --------------------------------------------------------
    #[test]
    fn test_multi_method_selection() {
        // let $dShow = (,,) show_impl showPrec_impl showList_impl
        // in ($sel_0 $dShow, $sel_2 $dShow)
        let dict_rhs = mk_tuple3(
            mk_var_expr("show_impl", 10),
            mk_var_expr("showPrec_impl", 11),
            mk_var_expr("showList_impl", 12),
        );
        // Use $sel_0 and $sel_2 in a pair
        let sel0 = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("$dShow", 50));
        let sel2 = mk_app(mk_var_expr("$sel_2", 0), mk_var_expr("$dShow", 50));
        let body = mk_tuple2(sel0, sel2);
        let expr = mk_let_nonrec("$dShow", 50, dict_rhs, body);

        let mut module = mk_module(vec![Bind::NonRec(mk_var("test", 100), Box::new(expr))]);
        let count = specialize_dictionaries(&mut module);

        assert_eq!(count, 2);
    }

    // --------------------------------------------------------
    // Test 8: No dict binding — $sel_N on unknown variable → unchanged
    // --------------------------------------------------------
    #[test]
    fn test_unknown_dict_unchanged() {
        // $sel_0 unknown_var
        let sel_call = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("unknown", 99));
        let mut module = mk_module(vec![Bind::NonRec(
            mk_var("test", 100),
            Box::new(sel_call),
        )]);

        let count = specialize_dictionaries(&mut module);
        assert_eq!(count, 0);
    }

    // --------------------------------------------------------
    // Test 9: Non-dict let — variable not starting with $d → skipped
    // --------------------------------------------------------
    #[test]
    fn test_non_dict_let_skipped() {
        // let x = (,) a b in $sel_0 x
        let tuple_rhs = mk_tuple2(mk_var_expr("a", 10), mk_var_expr("b", 11));
        let sel_call = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("x", 50));
        let expr = mk_let_nonrec("x", 50, tuple_rhs, sel_call);

        let mut module = mk_module(vec![Bind::NonRec(mk_var("test", 100), Box::new(expr))]);
        let count = specialize_dictionaries(&mut module);

        // "x" doesn't start with "$d", so not added to dict env
        assert_eq!(count, 0);
    }

    // --------------------------------------------------------
    // Test 10: Nested dicts — both used
    // --------------------------------------------------------
    #[test]
    fn test_nested_dicts() {
        // let $d1 = (,) m1 m2
        // in let $d2 = (,) m3 m4
        //    in ($sel_0 $d1, $sel_1 $d2)
        let d1_rhs = mk_tuple2(mk_var_expr("m1", 10), mk_var_expr("m2", 11));
        let d2_rhs = mk_tuple2(mk_var_expr("m3", 12), mk_var_expr("m4", 13));

        let sel_d1 = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("$d1", 50));
        let sel_d2 = mk_app(mk_var_expr("$sel_1", 0), mk_var_expr("$d2", 51));
        let inner_body = mk_tuple2(sel_d1, sel_d2);

        let inner = mk_let_nonrec("$d2", 51, d2_rhs, inner_body);
        let outer = mk_let_nonrec("$d1", 50, d1_rhs, inner);

        let mut module = mk_module(vec![Bind::NonRec(mk_var("test", 100), Box::new(outer))]);
        let count = specialize_dictionaries(&mut module);

        assert_eq!(count, 2);
    }

    // --------------------------------------------------------
    // Test 11: Direct tuple (not variable) — $sel_0 applied to tuple expr
    // --------------------------------------------------------
    #[test]
    fn test_direct_tuple_arg() {
        // $sel_0 ((,) a b) — arg is directly a tuple, not a variable
        let tuple = mk_tuple2(mk_var_expr("a", 10), mk_var_expr("b", 11));
        let sel_call = mk_app(mk_var_expr("$sel_0", 0), tuple);

        let mut module = mk_module(vec![Bind::NonRec(
            mk_var("test", 100),
            Box::new(sel_call),
        )]);
        let count = specialize_dictionaries(&mut module);

        assert_eq!(count, 1);

        // Result should be "a" (VarId 10)
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            assert!(
                is_var_with_id(rhs, 10),
                "expected a (id=10), got {:?}",
                rhs
            );
        }
    }

    // --------------------------------------------------------
    // Test 12: Recursive binding — dict specialization inside Rec
    // --------------------------------------------------------
    #[test]
    fn test_recursive_binding() {
        // rec { go = let $d = (,) m1 m2 in $sel_0 $d }
        let dict_rhs = mk_tuple2(mk_var_expr("m1", 10), mk_var_expr("m2", 11));
        let sel_call = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("$d", 50));
        let body = mk_let_nonrec("$d", 50, dict_rhs, sel_call);

        let mut module = mk_module(vec![Bind::Rec(vec![(mk_var("go", 100), Box::new(body))])]);
        let count = specialize_dictionaries(&mut module);

        assert_eq!(count, 1);
    }

    // --------------------------------------------------------
    // Test 13: Dict inside main body — still specialized
    // --------------------------------------------------------
    #[test]
    fn test_dict_inside_main() {
        // main = let $d = (,) f g in $sel_1 $d
        let dict_rhs = mk_tuple2(mk_var_expr("f", 10), mk_var_expr("g", 11));
        let sel_call = mk_app(mk_var_expr("$sel_1", 0), mk_var_expr("$d", 50));
        let body = mk_let_nonrec("$d", 50, dict_rhs, sel_call);

        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 100), Box::new(body))]);
        let count = specialize_dictionaries(&mut module);

        // Dict specialization works inside main — we traverse the body
        assert_eq!(count, 1);
    }

    // --------------------------------------------------------
    // Test 14: Superclass extraction — chained $sel_N
    // --------------------------------------------------------
    #[test]
    fn test_superclass_extraction() {
        // let $dOrd = (,,) $dEq method1 method2
        // in let $dEq_extracted = $sel_0 $dOrd   -- extracts superclass Eq dict
        //    in $sel_0 $dEq_extracted             -- extracts eq method from Eq dict
        //
        // But since $dEq_extracted is not a $d-prefixed let, we need to
        // set it up properly. Let's use $dEq prefix.
        let eq_dict = mk_tuple2(mk_var_expr("eq_fn", 10), mk_var_expr("ne_fn", 11));
        let ord_rhs = mk_tuple3(
            eq_dict,
            mk_var_expr("compare_fn", 12),
            mk_var_expr("lt_fn", 13),
        );

        // $sel_0 $dOrd → extracts the Eq dict (which is a tuple itself)
        let extract_eq = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("$dOrd", 50));

        // let $dEq_from_Ord = $sel_0 $dOrd
        // in $sel_0 $dEq_from_Ord
        let use_eq = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("$dEq_from_Ord", 51));
        let inner = mk_let_nonrec("$dEq_from_Ord", 51, extract_eq, use_eq);
        let outer = mk_let_nonrec("$dOrd", 50, ord_rhs, inner);

        let mut module = mk_module(vec![Bind::NonRec(mk_var("test", 100), Box::new(outer))]);
        let count = specialize_dictionaries(&mut module);

        // First $sel_0 on $dOrd fires (extracting Eq dict), then the extracted
        // value is a tuple, so the second $sel_0 fires too
        assert!(count >= 1, "expected at least 1 specialization, got {count}");
    }

    // --------------------------------------------------------
    // Test 15: Selection inside lambda body
    // --------------------------------------------------------
    #[test]
    fn test_selection_inside_lambda() {
        // \x -> let $d = (,) f g in $sel_0 $d
        let dict_rhs = mk_tuple2(mk_var_expr("f", 10), mk_var_expr("g", 11));
        let sel_call = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("$d", 50));
        let inner = mk_let_nonrec("$d", 50, dict_rhs, sel_call);
        let lambda = mk_lam("x", 1, inner);

        let mut module = mk_module(vec![Bind::NonRec(mk_var("test", 100), Box::new(lambda))]);
        let count = specialize_dictionaries(&mut module);

        assert_eq!(count, 1);
    }

    // --------------------------------------------------------
    // Test 16: Selection inside case alternatives
    // --------------------------------------------------------
    #[test]
    fn test_selection_inside_case() {
        // case x of { Default -> let $d = (,) f g in $sel_0 $d }
        let dict_rhs = mk_tuple2(mk_var_expr("f", 10), mk_var_expr("g", 11));
        let sel_call = mk_app(mk_var_expr("$sel_0", 0), mk_var_expr("$d", 50));
        let alt_body = mk_let_nonrec("$d", 50, dict_rhs, sel_call);

        let case_expr = Expr::Case(
            Box::new(mk_var_expr("x", 1)),
            vec![Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: alt_body,
            }],
            Ty::Error,
            Span::default(),
        );

        let mut module = mk_module(vec![Bind::NonRec(
            mk_var("test", 100),
            Box::new(case_expr),
        )]);
        let count = specialize_dictionaries(&mut module);

        assert_eq!(count, 1);
    }

    // --------------------------------------------------------
    // Test 17: Out-of-bounds $sel_N index → unchanged
    // --------------------------------------------------------
    #[test]
    fn test_out_of_bounds_index() {
        // let $d = (,) a b in $sel_5 $d — index 5 but only 2 fields
        let dict_rhs = mk_tuple2(mk_var_expr("a", 10), mk_var_expr("b", 11));
        let sel_call = mk_app(mk_var_expr("$sel_5", 0), mk_var_expr("$d", 50));
        let expr = mk_let_nonrec("$d", 50, dict_rhs, sel_call);

        let mut module = mk_module(vec![Bind::NonRec(mk_var("test", 100), Box::new(expr))]);
        let count = specialize_dictionaries(&mut module);

        assert_eq!(count, 0);
    }
}
