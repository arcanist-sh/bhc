//! List-fusion rewrites for the Numeric profile.
//!
//! These implement the "guaranteed fusion patterns" (H26-SPEC Section 8) as
//! semantics-preserving Core→Core rewrites. Because native codegen consumes Core
//! directly, fusing here removes intermediate list allocation on every backend
//! without touching codegen.
//!
//! Gated on [`SimplifyConfig::fuse_lists`](super::SimplifyConfig), which the
//! driver enables only for the Numeric profile.
//!
//! Currently implemented:
//! - **Pattern 1** — `map f (map g xs)` → `map (\v -> f (g v)) xs` (one traversal).
//!   The composed lambda is a beta-redex the simplifier reduces on the next pass.

use super::expr_util::fresh_var_id;
use crate::{Expr, Literal, Ty, Var};
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

/// Fuse `sum (map f xs)` into `foldl' (\acc x -> acc + f x) 0 xs` — a strict left
/// fold that accumulates `f x` without materializing the mapped list (Pattern 3).
///
/// Scoped to **integer** sums: only fires when `f`'s codomain is `Int`, so the
/// `0` literal and `+` are unambiguous. Returns `None` otherwise (non-`Int`
/// sums fall back to the unfused form, which is correct, just slower).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VarId;
    use bhc_types::{Kind, TyCon};

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
}
