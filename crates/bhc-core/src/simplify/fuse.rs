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
//! # Type dependence and known limitations (measured 2026-07-24 — re-measure before trusting)
//!
//! `try_fuse_sum_map` gates on `f`'s codomain being `Int` (so the `0` literal and
//! `+` are unambiguous). That gate reads [`Expr::ty`], which the typed-Core-IR
//! work (BHC-BRIEF-0002) now populates from typeck's per-node `expr_types`:
//!
//! 1. **`try_fuse_sum_map` fires for a *named* mapped function** (verified on
//!    both native and wasm). For `sum (map dbl xs)` where `dbl :: Int -> Int`,
//!    `f.ty()` is a real `Fun(Int, Int)`, the gate passes, and the rewrite to
//!    strict `foldl'` fires (the flagship compiles and prints the right answer).
//!    It still returns `None` for a **lambda**-mapped function
//!    (`map (\x -> x*2) xs`), where `f.ty()` is `Fun(Error, Error)`. Root cause
//!    (measured): for a lambda, `Expr::ty()` composes `Fun(param.ty, body.ty())`,
//!    and both stay `Error` — the parameter is a scalar `Int` that `annotate_ty`
//!    deliberately leaves alone (codegen infers scalar widths from `Ty::Error`),
//!    and `body = x * 2` cannot compose to `Int` because the `*` operator `Var`
//!    **shares the source span of the whole `x * 2` application**, so span-keyed
//!    typeck records that span's type as the application's *result* (`Int`, a
//!    `Con`) rather than the operator's function type. `annotate_ty` (Fun-only)
//!    then skips it and `body.ty()` is `Error`. Closing this needs operator-span
//!    disambiguation (or a `Lam`/`App` type slot) — not a leaf tweak, and it
//!    overlaps the codegen-consumes-types work; do not force it via scalar
//!    annotation.
//! 2. **`try_fuse_map_map` fires** (it is type-agnostic — it only needs `g`'s type
//!    to be *some* `Fun`), and it is correct, but it does **not** by itself make
//!    numeric loops fast: codegen boxes every `Int`, so the per-element boxing —
//!    not intermediate lists — dominates the flagship `sum (map f [1..N])`. An
//!    honest Numeric performance contract is blocked on **unboxed codegen** as
//!    well; that is out of scope here. Any perf claim about these rewrites must
//!    come from an isolated numeric-with-fusion vs numeric-without-fusion
//!    measurement, not a default-vs-numeric comparison (which also varies RTS/GC
//!    config).

use super::expr_util::fresh_var_id;
use super::subst::{substitute, substitute_single};
use crate::{Alt, AltCon, Bind, DataCon, Expr, Literal, Ty, Var, VarId};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Kind, TyCon};
use rustc_hash::FxHashMap;

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
/// **Fires when `f` is a named function** whose type reaches Core. The
/// typed-Core-IR work (BHC-BRIEF-0002) populates a function-typed `Var`'s slot
/// during lowering, so for `sum (map dbl xs)` with `dbl :: Int -> Int`, `f.ty()`
/// is a real `Fun(Int, Int)`, the `Int` gate passes, and the rewrite fires
/// (verified end-to-end). It still returns `None` for a **lambda**-mapped
/// function, whose parameter/operator types are not yet annotated — see the
/// module header. The unit tests below supply explicit `Int` types and so
/// exercise the rewrite regardless.
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
/// matched `fold (enumFromTo a b)` or `fold (map f (enumFromTo a b))`, where
/// `fold` is a strict `i64` reduction with binary primop `op` and identity `seed`
/// (`sum` → `("+", 0)`, `product` → `("*", 1)`).
///
/// `make_elem(i)` produces the value combined for index `i`: `i` itself for a
/// plain `fold (enumFromTo …)`, or `f` inlined at `i` (its body with the binder
/// substituted by `i`) for the `map` case. Returns `(go_var, go_lambda, replacement)`:
///
/// ```text
/// go = \b acc i -> case (i <= b) of
///                    True  -> go b (acc `op` make_elem(i)) (i + 1)
///                    False -> acc
/// ```
///
/// and `replacement` is `go b seed a`. `b` (the loop bound) is threaded as a
/// parameter so `go` is **closed** and can live at top level — bhc codegen only
/// turns *top-level* self-recursive functions into loops (a local `let rec`/
/// `where` helper does not bind its params: `stub: acc not implemented`).
fn build_fold_enum_loop(
    a: &Expr,
    b: &Expr,
    seed: Expr,
    combine: impl Fn(&Expr, Expr) -> Expr,
    make_elem: impl Fn(&Expr) -> Expr,
) -> (Var, Expr, Expr) {
    let go = fresh_var("$foldloop", Ty::Error);
    let bnd = fresh_var("$b", int_ty());
    let acc = fresh_var("$acc", int_ty());
    let i = fresh_var("$i", int_ty());

    let go_ref = || Expr::Var(go.clone(), Span::default());
    let b_ref = || Expr::Var(bnd.clone(), Span::default());
    let acc_ref = || Expr::Var(acc.clone(), Span::default());
    let i_ref = || Expr::Var(i.clone(), Span::default());

    // i <= b
    let cond = app2(builtin_var("<="), i_ref(), b_ref());
    // go b (combine acc make_elem(i)) (i + 1)  — a 3-argument application.
    // The counter step is always `+ 1`; only the accumulator uses `combine`.
    let next_acc = combine(&acc_ref(), make_elem(&i_ref()));
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
    // go b seed a
    let replacement = app3(go_ref(), b.clone(), seed, a.clone());
    (go, go_lam, replacement)
}

/// Fuse `<fold> (enumFromTo a b)` or `<fold> (map f (enumFromTo a b))` (f
/// manifestly `Int -> Int`) into a hoisted counting loop with initial accumulator
/// `seed` and per-step combiner `combine(acc, elem)`. Pushes the generated
/// top-level loop into `hoisted` and returns the replacement call, or `None` if
/// `list_arg` is neither an `enumFromTo` nor an Int-safe `map` over one.
fn fuse_fold_over_list(
    list_arg: &Expr,
    seed: Expr,
    combine: impl Fn(&Expr, Expr) -> Expr,
    hoisted: &mut Vec<Bind>,
) -> Option<Expr> {
    // map f (enumFromTo a b), f manifestly Int -> Int (inlined at the index).
    if let Some((_, f, inner)) = as_map_app(list_arg) {
        if let (Some((a, b)), Some((xid, fbody))) = (as_enum_from_to(inner), as_int_map_fn(f)) {
            let fbody = fbody.clone();
            let (go, go_lam, repl) = build_fold_enum_loop(a, b, seed, combine, |i| {
                substitute_single(fbody.clone(), xid, i)
            });
            hoisted.push(Bind::NonRec(go, Box::new(go_lam)));
            return Some(repl);
        }
        return None;
    }
    // plain enumFromTo a b.
    if let Some((a, b)) = as_enum_from_to(list_arg) {
        let (go, go_lam, repl) = build_fold_enum_loop(a, b, seed, combine, |i| i.clone());
        hoisted.push(Bind::NonRec(go, Box::new(go_lam)));
        return Some(repl);
    }
    None
}

/// A strict `i64` range reducer we can fuse: its `(op, seed)` where `op` is the
/// binary primop codegen lowers to `i64` and `seed` is the fold identity.
///
/// Both are airtight over `enumFromTo` (monomorphically `i64`): `sum` of an `i64`
/// range is an `i64` sum (empty → 0); `product` is an `i64` product (empty → 1).
fn fold_consumer(name: &str) -> Option<(&'static str, i64)> {
    match name {
        "sum" => Some(("+", 0)),
        "product" => Some(("*", 1)),
        _ => None,
    }
}

/// Match `foldl' op z list` = `App(App(App(Var("foldl'"), op), z), list)`.
fn as_foldl3(e: &Expr) -> Option<(&Expr, &Expr, &Expr)> {
    let (head, args) = app_spine(e);
    if let Expr::Var(v, _) = head {
        let base = v.name.as_str().rsplit('.').next().unwrap_or("");
        if (base == "foldl'" || base == "foldl") && args.len() == 3 {
            return Some((args[0], args[1], args[2]));
        }
    }
    None
}

/// Names of primops that are closed over `Int` (map `Int`/`Int,Int` to `Int`) and
/// that native codegen lowers directly to `i64` machine ops. Used to recognise a
/// map function that is *manifestly* `Int -> Int` without any type information.
///
/// Deliberately excludes `/` (fractional), `fromIntegral`/`realToFrac` (coercions),
/// and unary `Num` methods whose codegen dispatch is less certain.
fn is_int_arith_prim(name: &str) -> bool {
    // Accept both bare operators and their qualified forms (e.g. `GHC.Num.+`).
    let base = name.rsplit('.').next().unwrap_or(name);
    matches!(base, "+" | "-" | "*" | "div" | "mod" | "quot" | "rem")
}

/// Decompose an application into `(head, args)` (left-to-right).
fn app_spine(e: &Expr) -> (&Expr, Vec<&Expr>) {
    let mut args = Vec::new();
    let mut cur = e;
    while let Expr::App(f, a, _) = cur {
        args.push(a.as_ref());
        cur = f.as_ref();
    }
    args.reverse();
    (cur, args)
}

/// True if `e` computes an `Int` purely from the `allowed` binders, `Int`
/// literals, and [`is_int_arith_prim`] applications — hence, fed `i64`s for those
/// binders (from the loop), it yields an `i64`. Any other variable (unknown
/// function, captured value, potential coercion like `fromIntegral`), non-`Int`
/// literal, or non-arithmetic node makes it return `false`. Because the only
/// permitted variables are the `allowed` binders, a body that passes is also
/// **closed** (modulo name-dispatched primops), so it can be inlined into a
/// top-level loop.
fn is_int_arith_expr(e: &Expr, allowed: &[VarId]) -> bool {
    match e {
        Expr::Var(v, _) => allowed.contains(&v.id),
        Expr::Lit(Literal::Int(_), _, _) => true,
        Expr::App(..) => {
            let (head, args) = app_spine(e);
            !args.is_empty()
                && matches!(head, Expr::Var(v, _) if is_int_arith_prim(v.name.as_str()))
                && args.iter().all(|a| is_int_arith_expr(a, allowed))
        }
        _ => false,
    }
}

/// If `f` is a lambda `\x -> body` whose `body` is manifestly `Int -> Int`
/// ([`is_int_arith_expr`]), return `(binder_id, body)` so the caller can inline it
/// into the counting loop. Returns `None` for anything else (bare functions like
/// `fromIntegral`, multi-argument lambdas, captures, non-arithmetic bodies).
fn as_int_map_fn(f: &Expr) -> Option<(VarId, &Expr)> {
    if let Expr::Lam(x, body, _) = f {
        if is_int_arith_expr(body, &[x.id]) {
            return Some((x.id, body));
        }
    }
    None
}

/// If `op` is a curried two-argument lambda `\acc x -> body` whose `body` is
/// manifestly `Int`-arithmetic over both binders ([`is_int_arith_expr`]), return
/// `(acc_id, x_id, body)` so the caller can inline it as the fold combiner.
/// Returns `None` otherwise (captures, non-arithmetic, wrong arity).
fn as_int_fold_op(op: &Expr) -> Option<(VarId, VarId, &Expr)> {
    if let Expr::Lam(acc, inner, _) = op {
        if let Expr::Lam(x, body, _) = inner.as_ref() {
            if is_int_arith_expr(body, &[acc.id, x.id]) {
                return Some((acc.id, x.id, body));
            }
        }
    }
    None
}

/// Module-level fusion pass. Rewrites every strict `i64`-range reduction over an
/// `enumFromTo` into a call to a freshly-generated top-level counting loop,
/// appending those loops to the module. Handled reductions:
/// - `sum`/`product (enumFromTo a b)` (see `fold_consumer`);
/// - `foldl' op z (enumFromTo a b)` with a manifestly `Int`-arithmetic `op`
///   (`\acc x -> …`, see `as_int_fold_op`) and an `Int`-literal `z`;
/// - any of the above with an interposed manifestly-`Int -> Int` `map f`.
///
/// Returns the number of rewrites performed.
///
/// # Why this is safe without types (unlike [`try_fuse_sum_map`])
///
/// `enumFromTo` is **monomorphically `i64`** in native codegen (its loop counter
/// is `i64`, boxing via `int_to_ptr`; a `Double` range like `[1.0..3.0]` is a
/// different producer). So a `sum`/`product` over `enumFromTo a b` is always an
/// `i64` reduction, and the generated loop reproduces exactly that — seed the
/// `op` identity, combine with the `i64` `op`, compare with `i64 <=` —
/// independent of the erased Core types. See `fold_consumer`.
///
/// The `map` case is admitted **only** when `as_int_map_fn` proves `f` is
/// manifestly `Int -> Int` (its body is built solely from the binder, `Int`
/// literals, and `Int`-closed primops). This structurally excludes the one hazard
/// a type-free rewrite would otherwise hit — a type-changing map such as
/// `fromIntegral`, which would make the accumulator `Double`. `f` is inlined at
/// the loop index, so the loop stays a tight `i64` loop.
///
/// Measured: `sum [1..100M]` and `sum (map (*2) [1..100M])` drop from ~2.5s / ~9.8s
/// to ~0 (the top-level loop is TCO'd and closed-formed by LLVM).
pub fn fuse_fold_enum_module(module: &mut crate::CoreModule) -> usize {
    let mut hoisted: Vec<Bind> = Vec::new();
    let mut bindings = std::mem::take(&mut module.bindings);
    for bind in &mut bindings {
        match bind {
            Bind::NonRec(_, rhs) => rewrite_fold_enum(rhs, &mut hoisted),
            Bind::Rec(pairs) => {
                for (_, rhs) in pairs.iter_mut() {
                    rewrite_fold_enum(rhs, &mut hoisted);
                }
            }
        }
    }
    let count = hoisted.len();
    bindings.extend(hoisted);
    module.bindings = bindings;
    count
}

/// Recursively rewrite `sum`/`product` over `enumFromTo` sub-expressions in place,
/// pushing each generated top-level loop binding into `hoisted`.
fn rewrite_fold_enum(e: &mut Expr, hoisted: &mut Vec<Bind>) {
    // Recurse into children first (post-order), so nested matches are handled.
    match e {
        Expr::App(f, a, _) => {
            rewrite_fold_enum(f, hoisted);
            rewrite_fold_enum(a, hoisted);
        }
        Expr::TyApp(f, _, _) => rewrite_fold_enum(f, hoisted),
        Expr::Lam(_, body, _) | Expr::TyLam(_, body, _) => rewrite_fold_enum(body, hoisted),
        Expr::Let(bind, body, _) => {
            match bind.as_mut() {
                Bind::NonRec(_, rhs) => rewrite_fold_enum(rhs, hoisted),
                Bind::Rec(pairs) => {
                    for (_, rhs) in pairs.iter_mut() {
                        rewrite_fold_enum(rhs, hoisted);
                    }
                }
            }
            rewrite_fold_enum(body, hoisted);
        }
        Expr::Case(scrut, alts, _, _) => {
            rewrite_fold_enum(scrut, hoisted);
            for alt in alts.iter_mut() {
                rewrite_fold_enum(&mut alt.rhs, hoisted);
            }
        }
        Expr::Lazy(inner, _) | Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) => {
            rewrite_fold_enum(inner, hoisted);
        }
        Expr::Var(_, _) | Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
    }

    // (A) foldl' op z (list), op a manifestly Int-arithmetic `\acc x -> …` and
    // z an Int literal (so the accumulator is i64). Inline op as the combiner.
    if let Some((op_expr, z, list_arg)) = as_foldl3(e) {
        if let (Some((acc_id, x_id, op_body)), true) = (
            as_int_fold_op(op_expr),
            matches!(z, Expr::Lit(Literal::Int(_), _, _)),
        ) {
            let op_body = op_body.clone();
            let seed = z.clone();
            if let Some(repl) = fuse_fold_over_list(
                list_arg,
                seed,
                |acc, elem| {
                    let mut m = FxHashMap::default();
                    m.insert(acc_id, acc.clone());
                    m.insert(x_id, elem);
                    substitute(op_body.clone(), &m)
                },
                hoisted,
            ) {
                *e = repl;
            }
        }
        return;
    }

    // (B) `<consumer> <list_arg>` where consumer is a 1-arg i64-range reducer
    // (sum/product); combine with its named primop and identity seed.
    let Expr::App(head, list_arg, _) = e else {
        return;
    };
    let Expr::Var(cv, _) = head.as_ref() else {
        return;
    };
    let Some((op, seed)) = fold_consumer(cv.name.as_str()) else {
        return;
    };
    if let Some(repl) = fuse_fold_over_list(
        list_arg.as_ref(),
        int_lit(seed),
        |acc, elem| app2(builtin_var(op), acc.clone(), elem),
        hoisted,
    ) {
        *e = repl;
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
        assert_eq!(fuse_fold_enum_module(&mut m), 1);
        // main + one hoisted loop binding
        assert_eq!(m.bindings.len(), 2);
        // main's body is now a call whose head is the hoisted loop
        let Bind::NonRec(_, main_body) = &m.bindings[0] else {
            panic!("expected main NonRec");
        };
        assert!(head_name(main_body).unwrap().starts_with("$foldloop"));
        // the hoisted binding is a lambda named $sumloop*
        let Bind::NonRec(v, rhs) = &m.bindings[1] else {
            panic!("expected hoisted NonRec");
        };
        assert!(v.name.as_str().starts_with("$foldloop"));
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
        assert_eq!(fuse_fold_enum_module(&mut m), 0);
        assert_eq!(m.bindings.len(), 1);
    }

    fn enum_1_10() -> Expr {
        app2(var("enumFromTo", 30, Ty::Error), int_lit(1), int_lit(10))
    }

    /// `sum (map (\x -> <body>) (enumFromTo 1 10))` with the given lambda body.
    fn sum_map_enum(binder: Var, body: Expr) -> Expr {
        let f = Expr::Lam(binder, Box::new(body), Span::default());
        Expr::App(
            Box::new(var("sum", 42, Ty::Error)),
            Box::new(map_app(f, enum_1_10())),
            Span::default(),
        )
    }

    #[test]
    fn fuses_sum_map_enum_when_fn_is_int_arith() {
        // sum (map (\x -> x * 2) (enumFromTo 1 10)) — fires.
        let x = Var::new(Symbol::intern("x"), VarId::new(40), int_ty());
        let body = app2(
            var("*", 41, Ty::Error),
            Expr::Var(x.clone(), Span::default()),
            int_lit(2),
        );
        let mut m = one_binding_module(sum_map_enum(x, body));
        assert_eq!(fuse_fold_enum_module(&mut m), 1);
        assert_eq!(m.bindings.len(), 2);
        let Bind::NonRec(_, main_body) = &m.bindings[0] else {
            panic!("expected main NonRec");
        };
        assert!(head_name(main_body).unwrap().starts_with("$foldloop"));
    }

    #[test]
    fn does_not_fuse_sum_map_enum_unknown_fn() {
        // sum (map (\x -> g x) (enumFromTo 1 10)) — unknown `g` (could be a
        // coercion like fromIntegral) must decline.
        let x = Var::new(Symbol::intern("x"), VarId::new(40), int_ty());
        let body = Expr::App(
            Box::new(var("g", 41, Ty::Error)),
            Box::new(Expr::Var(x.clone(), Span::default())),
            Span::default(),
        );
        let mut m = one_binding_module(sum_map_enum(x, body));
        assert_eq!(fuse_fold_enum_module(&mut m), 0);
        assert_eq!(m.bindings.len(), 1);
    }

    #[test]
    fn does_not_fuse_sum_map_enum_free_capture() {
        // sum (map (\x -> x + k) (enumFromTo 1 10)) — `k` is not the binder, so
        // the body is not closed and must decline (inlining would free `k`).
        let x = Var::new(Symbol::intern("x"), VarId::new(40), int_ty());
        let body = app2(
            var("+", 41, Ty::Error),
            Expr::Var(x.clone(), Span::default()),
            var("k", 43, int_ty()),
        );
        let mut m = one_binding_module(sum_map_enum(x, body));
        assert_eq!(fuse_fold_enum_module(&mut m), 0);
        assert_eq!(m.bindings.len(), 1);
    }

    #[test]
    fn fuses_product_enum() {
        // product (enumFromTo 1 10) — fires with op `*`, seed 1.
        let body = Expr::App(
            Box::new(var("product", 50, Ty::Error)),
            Box::new(enum_1_10()),
            Span::default(),
        );
        let mut m = one_binding_module(body);
        assert_eq!(fuse_fold_enum_module(&mut m), 1);
        assert_eq!(m.bindings.len(), 2);
        let Bind::NonRec(_, main_body) = &m.bindings[0] else {
            panic!("expected main NonRec");
        };
        assert!(head_name(main_body).unwrap().starts_with("$foldloop"));
    }

    /// `foldl' (\acc x -> <op_body>) z (enumFromTo 1 10)`.
    fn foldl_enum(acc: Var, x: Var, op_body: Expr, z: Expr) -> Expr {
        let op = Expr::Lam(
            acc,
            Box::new(Expr::Lam(x, Box::new(op_body), Span::default())),
            Span::default(),
        );
        app3(var("foldl'", 60, Ty::Error), op, z, enum_1_10())
    }

    #[test]
    fn fuses_foldl_enum_int_op_and_literal_seed() {
        // foldl' (\acc x -> acc + x) 0 (enumFromTo 1 10) — fires.
        let acc = Var::new(Symbol::intern("acc"), VarId::new(61), int_ty());
        let x = Var::new(Symbol::intern("x"), VarId::new(62), int_ty());
        let op_body = app2(
            var("+", 63, Ty::Error),
            Expr::Var(acc.clone(), Span::default()),
            Expr::Var(x.clone(), Span::default()),
        );
        let mut m = one_binding_module(foldl_enum(acc, x, op_body, int_lit(0)));
        assert_eq!(fuse_fold_enum_module(&mut m), 1);
        assert_eq!(m.bindings.len(), 2);
        let Bind::NonRec(_, main_body) = &m.bindings[0] else {
            panic!("expected main NonRec");
        };
        assert!(head_name(main_body).unwrap().starts_with("$foldloop"));
    }

    #[test]
    fn does_not_fuse_foldl_enum_non_int_op() {
        // foldl' (\acc x -> g acc x) 0 (enumFromTo 1 10) — unknown `g` declines.
        let acc = Var::new(Symbol::intern("acc"), VarId::new(61), int_ty());
        let x = Var::new(Symbol::intern("x"), VarId::new(62), int_ty());
        let op_body = app2(
            var("g", 63, Ty::Error),
            Expr::Var(acc.clone(), Span::default()),
            Expr::Var(x.clone(), Span::default()),
        );
        let mut m = one_binding_module(foldl_enum(acc, x, op_body, int_lit(0)));
        assert_eq!(fuse_fold_enum_module(&mut m), 0);
        assert_eq!(m.bindings.len(), 1);
    }

    #[test]
    fn does_not_fuse_foldl_enum_non_literal_seed() {
        // foldl' (\acc x -> acc + x) z (enumFromTo 1 10) — non-literal seed `z`
        // (its type is unknown, so the accumulator may not be i64) declines.
        let acc = Var::new(Symbol::intern("acc"), VarId::new(61), int_ty());
        let x = Var::new(Symbol::intern("x"), VarId::new(62), int_ty());
        let op_body = app2(
            var("+", 63, Ty::Error),
            Expr::Var(acc.clone(), Span::default()),
            Expr::Var(x.clone(), Span::default()),
        );
        let z = var("z", 64, int_ty()); // a variable, not a Lit
        let mut m = one_binding_module(foldl_enum(acc, x, op_body, z));
        assert_eq!(fuse_fold_enum_module(&mut m), 0);
        assert_eq!(m.bindings.len(), 1);
    }
}
