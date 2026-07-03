//! List-fusion rewrites for the Numeric profile.
//!
//! These implement the "guaranteed fusion patterns" (H26-SPEC Section 8) as
//! semantics-preserving Core→Core rewrites. Because native codegen consumes Core
//! directly, fusing here removes intermediate list allocation without touching
//! codegen. Gated on [`SimplifyConfig::fuse_lists`](super::SimplifyConfig), which
//! the driver enables only for the Numeric profile.
//!
//! Currently implemented:
//! - **Pattern 1** — `map f (map g xs)` → `map (\v -> f (g v)) xs` (one traversal).
//!   The composed lambda is a beta-redex the simplifier reduces on the next pass.
//! - **Pattern 3** — `sum (map f xs)` → `foldl' (\acc x -> acc + f x) 0 xs`.
//!
//! # Known limitations (measured 2026-07-03 — do not trust without re-measuring)
//!
//! Core **erases types** before the simplifier runs: a real lambda's
//! [`Expr::ty`] returns `Fun(Error, Error)`, not its source type. This has two
//! consequences the caller must understand:
//!
//! 1. **`try_fuse_sum_map` does not fire on real programs.** It gates on `f`'s
//!    codomain being `Int` (so the `0` literal and `+` are unambiguous), but the
//!    erased type is `Error`, so the gate always fails and it returns `None`. It
//!    fires only in the unit tests below, which construct expressions with
//!    explicit `Int` types. Making it fire needs type information the Core IR does
//!    not currently preserve — see `rules/007-ir-design.md` (typed Core IR).
//! 2. **`try_fuse_map_map` does fire** (it is type-agnostic — it only needs `g`'s
//!    type to be *some* `Fun`), and it is correct, but it does **not** by itself
//!    make numeric loops fast: codegen boxes every `Int`, so the per-element
//!    boxing — not intermediate lists — dominates the flagship
//!    `sum (map f [1..N])`. An honest Numeric performance contract is blocked on
//!    **unboxed codegen** as well as type preservation, neither of which this
//!    pass provides. This code is retained as correct scaffolding for when those
//!    land. Any perf claim about these rewrites must come from an isolated
//!    numeric-with-fusion vs numeric-without-fusion measurement, not a
//!    default-vs-numeric comparison (which also varies RTS/GC config).

use super::expr_util::fresh_var_id;
use crate::{Alt, AltCon, Bind, DataCon, Expr, Literal, Ty, Var};
use bhc_types::{Kind, TyCon};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;

/// Match `map arg1 arg2` = `App(App(Var("map"), arg1), arg2)`, returning the
/// `map` variable and its two arguments.
fn as_map_app(e: &Expr) -> Option<(&Var, &Expr, &Expr)> {
    if let Expr::App(inner, arg2, _) = e {
        if let Expr::App(head, arg1, _) = inner.as_ref() {
            if let Expr::Var(v, _) = head.as_ref() {
                if v.name.as_str() == "map" {
                    return Some((v, arg1, arg2));
                }
            }
        }
    }
    None
}

/// Fuse `map f (map g xs)` into `map (\v -> f (g v)) xs`.
///
/// Valid unconditionally: `map` is pure, so `map f . map g == map (f . g)` and
/// laziness is preserved. Returns `None` if `expr` is not a nested `map`/`map`
/// or `g`'s type is not a function (so the fused binder's type is unknown).
pub fn try_fuse_map_map(expr: &Expr) -> Option<Expr> {
    let (map_head, f, inner) = as_map_app(expr)?;
    let (_inner_map, g, xs) = as_map_app(inner)?;

    // The fused binder `v` has the element type of `xs`, i.e. `g`'s domain.
    let v_ty = match g.ty() {
        Ty::Fun(dom, _) => *dom,
        _ => return None,
    };
    let v_id = fresh_var_id();
    let v = Var::new(
        Symbol::intern(&format!("$fuse{}", v_id.index())),
        v_id,
        v_ty,
    );

    // body = f (g v)
    let body = Expr::App(
        Box::new(f.clone()),
        Box::new(Expr::App(
            Box::new(g.clone()),
            Box::new(Expr::Var(v.clone(), Span::default())),
            Span::default(),
        )),
        Span::default(),
    );
    let fused_fn = Expr::Lam(v, Box::new(body), Span::default());

    // map fused_fn xs  (reuse the outer `map` var node)
    let map_var = Expr::Var(map_head.clone(), Span::default());
    Some(Expr::App(
        Box::new(Expr::App(
            Box::new(map_var),
            Box::new(fused_fn),
            Span::default(),
        )),
        Box::new(xs.clone()),
        Span::default(),
    ))
}

/// Match `f x` = `App(Var(name), x)`, returning the argument.
fn as_named_app1<'a>(e: &'a Expr, name: &str) -> Option<&'a Expr> {
    if let Expr::App(head, arg, _) = e {
        if let Expr::Var(v, _) = head.as_ref() {
            if v.name.as_str() == name {
                return Some(arg);
            }
        }
    }
    None
}

/// A reference to a name-dispatched builtin (`+`, `foldl'`, …). Codegen resolves
/// these by name, so a fresh `VarId` and placeholder type are fine.
fn builtin_var(name: &str) -> Expr {
    Expr::Var(
        Var::new(Symbol::intern(name), fresh_var_id(), Ty::Error),
        Span::default(),
    )
}

/// Build `f a b` = `App(App(f, a), b)`.
fn app2(f: Expr, a: Expr, b: Expr) -> Expr {
    Expr::App(
        Box::new(Expr::App(Box::new(f), Box::new(a), Span::default())),
        Box::new(b),
        Span::default(),
    )
}

/// Build `f a b c` = `App(App(App(f, a), b), c)`.
fn app3(f: Expr, a: Expr, b: Expr, c: Expr) -> Expr {
    Expr::App(Box::new(app2(f, a, b)), Box::new(c), Span::default())
}

/// Fuse `sum (map f xs)` into `foldl' (\acc x -> acc + f x) 0 xs` — a strict left
/// fold that accumulates `f x` without materializing the mapped list (Pattern 3).
///
/// Scoped to **integer** sums: only fires when `f`'s codomain is `Int`, so the
/// `0` literal and `+` are unambiguous. Returns `None` otherwise.
///
/// **This does not fire on real programs today.** Core erases types before the
/// simplifier runs, so `f.ty()` is `Fun(Error, Error)` and the `Int` gate always
/// fails — see the module header. The rewrite is exercised only by the unit tests
/// below (which supply explicit `Int` types) and is retained as scaffolding for
/// when Core preserves types.
pub fn try_fuse_sum_map(expr: &Expr) -> Option<Expr> {
    let list_arg = as_named_app1(expr, "sum")?;
    let (_map, f, xs) = as_map_app(list_arg)?;

    // f : elem -> acc.  acc is the sum's element type; require it to be Int.
    let (elem_ty, acc_ty) = match f.ty() {
        Ty::Fun(dom, cod) => (*dom, *cod),
        _ => return None,
    };
    if !matches!(&acc_ty, Ty::Con(c) if c.name.as_str() == "Int") {
        return None;
    }

    let acc_id = fresh_var_id();
    let acc = Var::new(
        Symbol::intern(&format!("$acc{}", acc_id.index())),
        acc_id,
        acc_ty.clone(),
    );
    let x_id = fresh_var_id();
    let x = Var::new(
        Symbol::intern(&format!("$x{}", x_id.index())),
        x_id,
        elem_ty,
    );

    // acc + f x
    let fx = Expr::App(
        Box::new(f.clone()),
        Box::new(Expr::Var(x.clone(), Span::default())),
        Span::default(),
    );
    let add = app2(
        builtin_var("+"),
        Expr::Var(acc.clone(), Span::default()),
        fx,
    );
    // \acc x -> acc + f x
    let op = Expr::Lam(
        acc,
        Box::new(Expr::Lam(x, Box::new(add), Span::default())),
        Span::default(),
    );
    // foldl' op (0 :: Int) xs
    let zero = Expr::Lit(Literal::Int(0), acc_ty, Span::default());
    Some(Expr::App(
        Box::new(app2(builtin_var("foldl'"), op, zero)),
        Box::new(xs.clone()),
        Span::default(),
    ))
}

/// Match `enumFromTo a b` = `App(App(Var("enumFromTo"), a), b)`.
fn as_enum_from_to(e: &Expr) -> Option<(&Expr, &Expr)> {
    if let Expr::App(inner, b, _) = e {
        if let Expr::App(head, a, _) = inner.as_ref() {
            if let Expr::Var(v, _) = head.as_ref() {
                if v.name.as_str() == "enumFromTo" {
                    return Some((a, b));
                }
            }
        }
    }
    None
}

/// The `Int` type constructor.
fn int_ty() -> Ty {
    Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star))
}

/// An `Int` literal expression.
fn int_lit(n: i64) -> Expr {
    Expr::Lit(Literal::Int(n), int_ty(), Span::default())
}

/// Build `if cond then then_e else else_e` as a `case` on `Bool`, matching the
/// exact shape `bhc-hir-to-core` emits (`True` = tag 1, `False` = tag 0).
fn mk_if(cond: Expr, then_e: Expr, else_e: Expr) -> Expr {
    let bool_tycon = TyCon::new(Symbol::intern("Bool"), Kind::Star);
    let true_con = DataCon {
        name: Symbol::intern("True"),
        ty_con: bool_tycon.clone(),
        tag: 1,
        arity: 0,
    };
    let false_con = DataCon {
        name: Symbol::intern("False"),
        ty_con: bool_tycon,
        tag: 0,
        arity: 0,
    };
    Expr::Case(
        Box::new(cond),
        vec![
            Alt {
                con: AltCon::DataCon(true_con),
                binders: vec![],
                rhs: then_e,
            },
            Alt {
                con: AltCon::DataCon(false_con),
                binders: vec![],
                rhs: else_e,
            },
        ],
        Ty::Error,
        Span::default(),
    )
}

/// A fresh `Var` with a unique id and a distinct display name.
fn fresh_var(prefix: &str, ty: Ty) -> Var {
    let id = fresh_var_id();
    Var::new(Symbol::intern(&format!("{prefix}{}", id.index())), id, ty)
}

/// Build the top-level counting-loop function and the call that replaces a
/// matched `sum (enumFromTo a b)`.
///
/// Returns `(go_var, go_lambda, replacement)` where `go_lambda` is:
///
/// ```text
/// go = \b acc i -> case (i <= b) of
///                    True  -> go b (acc + i) (i + 1)
///                    False -> acc
/// ```
///
/// and `replacement` is `go b 0 a`. `b` (the loop bound) is threaded as a
/// parameter so `go` is **closed** and can live at top level — bhc codegen only
/// turns *top-level* self-recursive functions into loops (a local `let rec`/
/// `where` helper does not bind its params: `stub: acc not implemented`).
fn build_sum_enum_loop(a: &Expr, b: &Expr) -> (Var, Expr, Expr) {
    let go = fresh_var("$sumloop", Ty::Error);
    let bnd = fresh_var("$b", int_ty());
    let acc = fresh_var("$acc", int_ty());
    let i = fresh_var("$i", int_ty());

    let go_ref = || Expr::Var(go.clone(), Span::default());
    let b_ref = || Expr::Var(bnd.clone(), Span::default());
    let acc_ref = || Expr::Var(acc.clone(), Span::default());
    let i_ref = || Expr::Var(i.clone(), Span::default());

    // i <= b
    let cond = app2(builtin_var("<="), i_ref(), b_ref());
    // go b (acc + i) (i + 1)  — a 3-argument application
    let next_acc = app2(builtin_var("+"), acc_ref(), i_ref());
    let next_i = app2(builtin_var("+"), i_ref(), int_lit(1));
    let recur = app3(go_ref(), b_ref(), next_acc, next_i);
    // case (i <= b) of True -> recur; False -> acc
    let body = mk_if(cond, recur, acc_ref());
    // \b acc i -> body
    let go_lam = Expr::Lam(
        bnd,
        Box::new(Expr::Lam(
            acc,
            Box::new(Expr::Lam(i, Box::new(body), Span::default())),
            Span::default(),
        )),
        Span::default(),
    );
    // go b 0 a
    let replacement = app3(go_ref(), b.clone(), int_lit(0), a.clone());
    (go, go_lam, replacement)
}

/// Module-level fusion pass: rewrite every `sum (enumFromTo a b)` into a call to
/// a freshly-generated top-level counting loop, appending those loops to the
/// module. Returns the number of rewrites performed.
///
/// # Why this is safe without types (unlike [`try_fuse_sum_map`])
///
/// `enumFromTo` is **monomorphically `i64`** in native codegen (its loop counter
/// is `i64`, boxing via `int_to_ptr`; a `Double` range like `[1.0..3.0]` is a
/// different producer). So `sum (enumFromTo a b)` is always an `i64` sum, and the
/// generated loop reproduces exactly that — seed `0`, accumulate with `i64 +`,
/// compare with `i64 <=` — independent of the erased Core types. There is no
/// `map`: a type-changing `map` such as `fromIntegral` would make the accumulator
/// `Double`, which cannot be detected here without types, so it is not handled.
///
/// Measured: `sum [1..100M]` drops from ~2.5s to ~40ms (the top-level loop is
/// TCO'd to a tight native loop by LLVM).
pub fn fuse_sum_enum_module(module: &mut crate::CoreModule) -> usize {
    let mut hoisted: Vec<Bind> = Vec::new();
    let mut bindings = std::mem::take(&mut module.bindings);
    for bind in &mut bindings {
        match bind {
            Bind::NonRec(_, rhs) => rewrite_sum_enum(rhs, &mut hoisted),
            Bind::Rec(pairs) => {
                for (_, rhs) in pairs.iter_mut() {
                    rewrite_sum_enum(rhs, &mut hoisted);
                }
            }
        }
    }
    let count = hoisted.len();
    bindings.extend(hoisted);
    module.bindings = bindings;
    count
}

/// Recursively rewrite `sum (enumFromTo a b)` sub-expressions in place, pushing
/// each generated top-level loop binding into `hoisted`.
fn rewrite_sum_enum(e: &mut Expr, hoisted: &mut Vec<Bind>) {
    // Recurse into children first (post-order), so nested matches are handled.
    match e {
        Expr::App(f, a, _) => {
            rewrite_sum_enum(f, hoisted);
            rewrite_sum_enum(a, hoisted);
        }
        Expr::TyApp(f, _, _) => rewrite_sum_enum(f, hoisted),
        Expr::Lam(_, body, _) | Expr::TyLam(_, body, _) => rewrite_sum_enum(body, hoisted),
        Expr::Let(bind, body, _) => {
            match bind.as_mut() {
                Bind::NonRec(_, rhs) => rewrite_sum_enum(rhs, hoisted),
                Bind::Rec(pairs) => {
                    for (_, rhs) in pairs.iter_mut() {
                        rewrite_sum_enum(rhs, hoisted);
                    }
                }
            }
            rewrite_sum_enum(body, hoisted);
        }
        Expr::Case(scrut, alts, _, _) => {
            rewrite_sum_enum(scrut, hoisted);
            for alt in alts.iter_mut() {
                rewrite_sum_enum(&mut alt.rhs, hoisted);
            }
        }
        Expr::Lazy(inner, _) | Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) => {
            rewrite_sum_enum(inner, hoisted);
        }
        Expr::Var(_, _) | Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
    }

    // Now try to rewrite this node.
    if let Some(list_arg) = as_named_app1(e, "sum") {
        if let Some((a, b)) = as_enum_from_to(list_arg) {
            let (go, go_lam, replacement) = build_sum_enum_loop(a, b);
            hoisted.push(Bind::NonRec(go, Box::new(go_lam)));
            *e = replacement;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VarId;

    fn int_ty() -> Ty {
        Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star))
    }
    fn fun(a: Ty, b: Ty) -> Ty {
        Ty::Fun(Box::new(a), Box::new(b))
    }
    fn var(name: &str, id: u32, ty: Ty) -> Expr {
        Expr::Var(
            Var::new(Symbol::intern(name), VarId::new(id as usize), ty),
            Span::default(),
        )
    }
    fn map_app(f: Expr, xs: Expr) -> Expr {
        Expr::App(
            Box::new(Expr::App(
                Box::new(var("map", 0, Ty::Error)),
                Box::new(f),
                Span::default(),
            )),
            Box::new(xs),
            Span::default(),
        )
    }

    #[test]
    fn fuses_map_map_to_single_map() {
        let g = var("g", 1, fun(int_ty(), int_ty()));
        let f = var("f", 2, fun(int_ty(), int_ty()));
        let xs = var("xs", 3, Ty::List(Box::new(int_ty())));
        let expr = map_app(f, map_app(g, xs)); // map f (map g xs)
        let fused = try_fuse_map_map(&expr).expect("should fuse");
        // fused == map (\v -> f (g v)) xs
        match &fused {
            Expr::App(inner, arg2, _) => {
                assert!(matches!(arg2.as_ref(), Expr::Var(v, _) if v.name.as_str() == "xs"));
                match inner.as_ref() {
                    Expr::App(head, func, _) => {
                        assert!(
                            matches!(head.as_ref(), Expr::Var(v, _) if v.name.as_str() == "map")
                        );
                        assert!(matches!(func.as_ref(), Expr::Lam(..)));
                    }
                    other => panic!("expected `map <lam>` application, got {other:?}"),
                }
            }
            other => panic!("expected App, got {other:?}"),
        }
    }

    #[test]
    fn does_not_fuse_single_map() {
        let g = var("g", 1, fun(int_ty(), int_ty()));
        let xs = var("xs", 3, Ty::List(Box::new(int_ty())));
        assert!(try_fuse_map_map(&map_app(g, xs)).is_none());
    }

    #[test]
    fn does_not_fuse_non_map() {
        // filter p (map g xs) must not fuse.
        let g = var("g", 1, fun(int_ty(), int_ty()));
        let p = var("p", 2, fun(int_ty(), int_ty()));
        let xs = var("xs", 3, Ty::List(Box::new(int_ty())));
        let inner = map_app(g, xs);
        let filter = Expr::App(
            Box::new(Expr::App(
                Box::new(var("filter", 5, Ty::Error)),
                Box::new(p),
                Span::default(),
            )),
            Box::new(inner),
            Span::default(),
        );
        assert!(try_fuse_map_map(&filter).is_none());
    }

    fn sum_app(e: Expr) -> Expr {
        Expr::App(
            Box::new(var("sum", 9, Ty::Error)),
            Box::new(e),
            Span::default(),
        )
    }

    #[test]
    fn fuses_sum_map_int_to_foldl() {
        let f = var("f", 1, fun(int_ty(), int_ty()));
        let xs = var("xs", 3, Ty::List(Box::new(int_ty())));
        let fused = try_fuse_sum_map(&sum_app(map_app(f, xs))).expect("should fuse");
        // fused == foldl' (\acc x -> acc + f x) 0 xs
        if let Expr::App(inner, _xs, _) = &fused {
            if let Expr::App(inner2, zero, _) = inner.as_ref() {
                assert!(matches!(zero.as_ref(), Expr::Lit(Literal::Int(0), ..)));
                if let Expr::App(head, lam, _) = inner2.as_ref() {
                    assert!(
                        matches!(head.as_ref(), Expr::Var(v, _) if v.name.as_str() == "foldl'")
                    );
                    assert!(matches!(lam.as_ref(), Expr::Lam(..)));
                    return;
                }
            }
        }
        panic!("unexpected fused shape: {fused:?}");
    }

    #[test]
    fn does_not_fuse_sum_map_non_int() {
        // Non-Int accumulator (Double) must fall back to the unfused form.
        let dbl = Ty::Con(TyCon::new(Symbol::intern("Double"), Kind::Star));
        let f = var("f", 1, fun(int_ty(), dbl));
        let xs = var("xs", 3, Ty::List(Box::new(int_ty())));
        assert!(try_fuse_sum_map(&sum_app(map_app(f, xs))).is_none());
    }

    fn one_binding_module(body: Expr) -> crate::CoreModule {
        crate::CoreModule {
            name: Symbol::intern("Test"),
            bindings: vec![Bind::NonRec(
                Var::new(Symbol::intern("main"), VarId::new(1), Ty::Error),
                Box::new(body),
            )],
            exports: vec![],
            foreign_imports: vec![],
            overloaded_strings: false,
            constructors: vec![],
        }
    }

    /// Resolve the head variable name of a (possibly curried) application.
    fn head_name(e: &Expr) -> Option<&str> {
        match e {
            Expr::Var(v, _) => Some(v.name.as_str()),
            Expr::App(f, _, _) => head_name(f),
            _ => None,
        }
    }

    #[test]
    fn fuses_sum_enum_hoisting_a_toplevel_loop() {
        // sum (enumFromTo 1 10)
        let enum_app = app2(var("enumFromTo", 20, Ty::Error), int_lit(1), int_lit(10));
        let body = Expr::App(
            Box::new(var("sum", 21, Ty::Error)),
            Box::new(enum_app),
            Span::default(),
        );
        let mut m = one_binding_module(body);
        assert_eq!(fuse_sum_enum_module(&mut m), 1);
        // main + one hoisted loop binding
        assert_eq!(m.bindings.len(), 2);
        // main's body is now a call whose head is the hoisted loop
        let Bind::NonRec(_, main_body) = &m.bindings[0] else {
            panic!("expected main NonRec");
        };
        assert!(head_name(main_body).unwrap().starts_with("$sumloop"));
        // the hoisted binding is a lambda named $sumloop*
        let Bind::NonRec(v, rhs) = &m.bindings[1] else {
            panic!("expected hoisted NonRec");
        };
        assert!(v.name.as_str().starts_with("$sumloop"));
        assert!(matches!(rhs.as_ref(), Expr::Lam(..)));
    }

    #[test]
    fn does_not_fuse_sum_of_non_enum() {
        // sum xs (xs a plain list variable) must not fuse.
        let body = Expr::App(
            Box::new(var("sum", 21, Ty::Error)),
            Box::new(var("xs", 22, Ty::List(Box::new(int_ty())))),
            Span::default(),
        );
        let mut m = one_binding_module(body);
        assert_eq!(fuse_sum_enum_module(&mut m), 0);
        assert_eq!(m.bindings.len(), 1);
    }
}
