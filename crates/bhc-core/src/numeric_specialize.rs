//! Specialize class-polymorphic functions at representation-sensitive types.
//!
//! BHC passes no dictionaries for the builtin classes (`Num`, `Eq`, `Ord`, …):
//! codegen picks an operation's implementation from its operands' TYPE. A
//! polymorphic body is lowered once, at its type variables, so
//!
//! ```haskell
//! sq :: Num a => a -> a
//! sq x = x * x
//! ```
//!
//! multiplies machine ints whatever `a` is — right at `Int` (the default
//! representation), wrong at `Integer` (a boxed bignum), `Double`, or a list
//! compared with `==` (#57).
//!
//! This pass gives such a binding a specialized copy for each concrete type it
//! is used at, with the type variable substituted throughout the copy (binder,
//! occurrence and literal types). Codegen's type-directed dispatch then picks
//! the right operations in the copy. Occurrences are redirected by typeck's
//! span-keyed `resolved_expr_types`, as in `monomorphize`.
//!
//! Scope: module-local, top-level bindings quantified over ONE type variable
//! (so every type variable in the body denotes it), whose bodies use a
//! class-sensitive builtin, used at `Integer`, `Double`, `Float`, `Rational`,
//! `Text` or a list type. `Int` and the other machine-int types need no copy.
//! No arity changes: the original binding stays as it is.

use crate::monomorphize::{collect_tyvars, has_free_tyvar, match_ty, ultimate_result};
use crate::simplify::expr_util::fresh_var_id;
use crate::{Alt, Bind, CoreModule, Expr, Literal, Var, VarId};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Subst, Ty, TyVar};
use rustc_hash::FxHashMap;

/// Most copies made of any one binding (bounds polymorphic recursion).
const MAX_SPECS_PER_BINDING: usize = 16;

/// Specialize class-polymorphic bindings at representation-sensitive types.
/// Returns the number of copies created. `BHC_NO_NUMSPEC=1` disables it.
pub fn specialize_numeric(module: &mut CoreModule, resolved: &FxHashMap<Span, Ty>) -> usize {
    if std::env::var_os("BHC_NO_NUMSPEC").is_some() {
        return 0;
    }
    let mut top: FxHashMap<VarId, (Var, Expr)> = FxHashMap::default();
    for bind in &module.bindings {
        match bind {
            Bind::NonRec(v, e) => {
                top.insert(v.id, (v.clone(), (**e).clone()));
            }
            Bind::Rec(bs) => {
                for (v, e) in bs {
                    top.insert(v.id, (v.clone(), (**e).clone()));
                }
            }
        }
    }
    let mut ctx = Ctx {
        top,
        resolved,
        memo: FxHashMap::default(),
        counts: FxHashMap::default(),
        new_bindings: Vec::new(),
    };
    let mut bindings = std::mem::take(&mut module.bindings);
    for bind in &mut bindings {
        match bind {
            Bind::NonRec(_, e) => redirect(e, None, &mut ctx),
            Bind::Rec(bs) => {
                for (_, e) in bs {
                    redirect(e, None, &mut ctx);
                }
            }
        }
    }
    let created = ctx.new_bindings.len();
    bindings.append(&mut ctx.new_bindings);
    module.bindings = bindings;
    created
}

struct Ctx<'a> {
    top: FxHashMap<VarId, (Var, Expr)>,
    resolved: &'a FxHashMap<Span, Ty>,
    /// (binder id, concrete type) -> specialized binder.
    memo: FxHashMap<(VarId, String), Var>,
    counts: FxHashMap<VarId, usize>,
    new_bindings: Vec<Bind>,
}

/// Redirect occurrences in `e`. `subst` is `Some` inside a copy being built:
/// occurrence types there are the substituted `Var.ty`s, since the span-keyed
/// map holds the generic body's types.
fn redirect(e: &mut Expr, subst: Option<&Subst>, ctx: &mut Ctx) {
    match e {
        Expr::Var(v, span) => {
            let occ = match subst {
                Some(s) => Some(s.apply(&v.ty)),
                None => ctx.resolved.get(span).cloned(),
            };
            if let Some(occ) = occ.filter(|t| !has_free_tyvar(t)) {
                if let Some(spec) = specialize_at(v, &occ, ctx) {
                    *e = Expr::Var(spec, *span);
                }
            }
        }
        Expr::App(f, x, _) => {
            redirect(f, subst, ctx);
            redirect(x, subst, ctx);
        }
        Expr::TyApp(f, _, _)
        | Expr::Lam(_, f, _)
        | Expr::TyLam(_, f, _)
        | Expr::Lazy(f, _)
        | Expr::Cast(f, _, _)
        | Expr::Tick(_, f, _) => redirect(f, subst, ctx),
        Expr::Let(bind, body, _) => {
            match bind.as_mut() {
                Bind::NonRec(_, r) => redirect(r, subst, ctx),
                Bind::Rec(bs) => {
                    for (_, r) in bs {
                        redirect(r, subst, ctx);
                    }
                }
            }
            redirect(body, subst, ctx);
        }
        Expr::Case(s, alts, _, _) => {
            redirect(s, subst, ctx);
            for a in alts {
                redirect(&mut a.rhs, subst, ctx);
            }
        }
        Expr::Lit(..) | Expr::Type(..) | Expr::Coercion(..) => {}
    }
}

/// The specialized copy of `v`'s binding for the concrete occurrence type
/// `occ`, when it needs one.
fn specialize_at(v: &Var, occ: &Ty, ctx: &mut Ctx) -> Option<Var> {
    let (orig, body) = ctx
        .top
        .get(&v.id)
        .filter(|(o, _)| o.name == v.name)?
        .clone();
    let mut vars: Vec<TyVar> = Vec::new();
    collect_tyvars(&orig.ty, &mut vars);
    let [tv] = vars.as_slice() else { return None };
    let mut m = Subst::new();
    if !(match_ty(&orig.ty, occ, &mut m) || {
        m = Subst::new();
        match_ty(ultimate_result(&orig.ty), ultimate_result(occ), &mut m)
    }) {
        return None;
    }
    let image = m.get(tv)?.clone();
    if has_free_tyvar(&image) || !needs_copy(&image) || !uses_class_builtin(&body) {
        return None;
    }
    let key = (orig.id, format!("{image:?}"));
    if let Some(spec) = ctx.memo.get(&key) {
        return Some(spec.clone());
    }
    let count = ctx.counts.entry(orig.id).or_insert(0);
    if *count >= MAX_SPECS_PER_BINDING {
        return None;
    }
    *count += 1;

    // Every type variable of the body denotes `tv` (one quantified variable;
    // Core occurrences carry their own fresh copies of it).
    let mut body_vars = Vec::new();
    collect_body_tyvars(&body, &mut body_vars);
    let mut subst = Subst::new();
    for bv in body_vars {
        subst.insert(&bv, image.clone());
    }
    subst.insert(tv, image.clone());

    let id = fresh_var_id();
    let name = Symbol::intern(&format!("{}$$num{}", orig.name.as_str(), id.index()));
    let spec = Var::new(name, id, subst.apply(&orig.ty));
    // Memoize before recursing, so a recursive body calls this same copy.
    ctx.memo.insert(key, spec.clone());
    let mut copy = substitute(&body, &subst);
    propagate(&mut copy, &mut FxHashMap::default());
    redirect(&mut copy, Some(&subst), ctx);
    // `BHC_DUMP_NUMSPEC=<substring>` dumps matching copies with their types.
    if let Ok(filter) = std::env::var("BHC_DUMP_NUMSPEC") {
        if orig.name.as_str().contains(filter.as_str()) {
            eprintln!(
                "=== NUMSPEC {} :: {:?} ===\n{:#?}",
                name.as_str(),
                spec.ty,
                copy
            );
        }
    }
    ctx.new_bindings
        .push(Bind::NonRec(spec.clone(), Box::new(copy)));
    Some(spec)
}

/// Types whose representation differs from the machine-int default, where a
/// class method's implementation depends on the type.
fn needs_copy(ty: &Ty) -> bool {
    match ty {
        Ty::Con(c) => matches!(
            c.name.as_str(),
            "Integer" | "Double" | "Float" | "Rational" | "Text" | "String"
        ),
        Ty::List(_) => true,
        Ty::App(f, _) => matches!(f.as_ref(), Ty::Con(c) if c.name.as_str() == "Ratio"),
        Ty::Tuple(ts) => ts.iter().any(needs_copy),
        _ => false,
    }
}

/// Whether `e` uses a builtin whose implementation depends on its operands'
/// type (arithmetic, comparison, numeric conversion, `show`, …).
fn uses_class_builtin(e: &Expr) -> bool {
    match e {
        Expr::Var(v, _) => {
            let name = v.name.as_str();
            let bare = name.rsplit('.').next().unwrap_or(name);
            matches!(
                bare,
                "+" | "-"
                    | "*"
                    | "/"
                    | "^"
                    | "negate"
                    | "abs"
                    | "signum"
                    | "fromInteger"
                    | "fromIntegral"
                    | "toInteger"
                    | "realToFrac"
                    | "div"
                    | "mod"
                    | "quot"
                    | "rem"
                    | "=="
                    | "/="
                    | "<"
                    | "<="
                    | ">"
                    | ">="
                    | "compare"
                    | "max"
                    | "min"
                    | "maximum"
                    | "minimum"
                    | "sum"
                    | "product"
                    | "elem"
                    | "notElem"
                    | "lookup"
                    | "nub"
                    | "sort"
                    | "show"
                    | "even"
                    | "odd"
            )
        }
        Expr::App(f, x, _) => uses_class_builtin(f) || uses_class_builtin(x),
        Expr::TyApp(f, _, _)
        | Expr::Lam(_, f, _)
        | Expr::TyLam(_, f, _)
        | Expr::Lazy(f, _)
        | Expr::Cast(f, _, _)
        | Expr::Tick(_, f, _) => uses_class_builtin(f),
        Expr::Let(bind, body, _) => {
            let in_bind = match bind.as_ref() {
                Bind::NonRec(_, r) => uses_class_builtin(r),
                Bind::Rec(bs) => bs.iter().any(|(_, r)| uses_class_builtin(r)),
            };
            in_bind || uses_class_builtin(body)
        }
        Expr::Case(s, alts, _, _) => {
            uses_class_builtin(s) || alts.iter().any(|a| uses_class_builtin(&a.rhs))
        }
        Expr::Lit(..) | Expr::Type(..) | Expr::Coercion(..) => false,
    }
}

/// Every type variable mentioned in the body's binder, occurrence and
/// literal types.
fn collect_body_tyvars(e: &Expr, out: &mut Vec<TyVar>) {
    let var = |v: &Var, out: &mut Vec<TyVar>| collect_tyvars(&v.ty, out);
    match e {
        Expr::Var(v, _) => var(v, out),
        Expr::Lit(_, t, _) => collect_tyvars(t, out),
        Expr::App(f, x, _) => {
            collect_body_tyvars(f, out);
            collect_body_tyvars(x, out);
        }
        Expr::Lam(v, b, _) => {
            var(v, out);
            collect_body_tyvars(b, out);
        }
        Expr::TyApp(f, _, _)
        | Expr::TyLam(_, f, _)
        | Expr::Lazy(f, _)
        | Expr::Cast(f, _, _)
        | Expr::Tick(_, f, _) => collect_body_tyvars(f, out),
        Expr::Let(bind, body, _) => {
            match bind.as_ref() {
                Bind::NonRec(v, r) => {
                    var(v, out);
                    collect_body_tyvars(r, out);
                }
                Bind::Rec(bs) => {
                    for (v, r) in bs {
                        var(v, out);
                        collect_body_tyvars(r, out);
                    }
                }
            }
            collect_body_tyvars(body, out);
        }
        Expr::Case(s, alts, t, _) => {
            collect_tyvars(t, out);
            collect_body_tyvars(s, out);
            for a in alts {
                for b in &a.binders {
                    var(b, out);
                }
                collect_body_tyvars(&a.rhs, out);
            }
        }
        Expr::Type(..) | Expr::Coercion(..) => {}
    }
}

/// `e` with `subst` applied to every binder, occurrence and literal type. An
/// integer literal whose type becomes `Integer` or a floating type becomes a
/// literal of that type (the generic body built a machine-int literal).
fn substitute(e: &Expr, subst: &Subst) -> Expr {
    let var = |v: &Var| Var::new(v.name, v.id, subst.apply(&v.ty));
    match e {
        Expr::Var(v, s) => Expr::Var(var(v), *s),
        Expr::Lit(lit, t, s) => {
            let t2 = subst.apply(t);
            let lit2 = match (lit, &t2) {
                (Literal::Int(n), Ty::Con(c)) if c.name.as_str() == "Integer" => {
                    Literal::Integer(i128::from(*n))
                }
                (Literal::Int(n), Ty::Con(c)) if matches!(c.name.as_str(), "Double" | "Float") =>
                {
                    #[allow(clippy::cast_precision_loss)]
                    Literal::Double(*n as f64)
                }
                _ => lit.clone(),
            };
            Expr::Lit(lit2, t2, *s)
        }
        Expr::App(f, x, s) => {
            // `fromInteger 3` in the copy: when the conversion now targets
            // `Integer` or a floating type and its argument is a machine
            // literal (the generic body's), emit the literal at that type;
            // `fromInteger` is the identity, so a machine `3` reached BigInt
            // arithmetic as a raw word.
            if let (Expr::Var(fv, _), Expr::Lit(Literal::Int(n), _, ls)) = (f.as_ref(), x.as_ref())
            {
                let bare = fv.name.as_str().rsplit('.').next().unwrap_or("");
                if matches!(bare, "fromInteger" | "fromIntegral") {
                    if let Ty::Fun(_, r) = subst.apply(&fv.ty) {
                        if let Ty::Con(c) = r.as_ref() {
                            match c.name.as_str() {
                                "Integer" => {
                                    return Expr::Lit(Literal::Integer(i128::from(*n)), *r, *ls)
                                }
                                "Double" | "Float" => {
                                    #[allow(clippy::cast_precision_loss)]
                                    return Expr::Lit(Literal::Double(*n as f64), *r, *ls);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            Expr::App(
                Box::new(substitute(f, subst)),
                Box::new(substitute(x, subst)),
                *s,
            )
        }
        Expr::TyApp(f, t, s) => Expr::TyApp(Box::new(substitute(f, subst)), subst.apply(t), *s),
        Expr::Lam(v, b, s) => Expr::Lam(var(v), Box::new(substitute(b, subst)), *s),
        Expr::TyLam(v, b, s) => Expr::TyLam(v.clone(), Box::new(substitute(b, subst)), *s),
        Expr::Lazy(b, s) => Expr::Lazy(Box::new(substitute(b, subst)), *s),
        Expr::Cast(b, c, s) => Expr::Cast(Box::new(substitute(b, subst)), c.clone(), *s),
        Expr::Tick(t, b, s) => Expr::Tick(t.clone(), Box::new(substitute(b, subst)), *s),
        Expr::Let(bind, body, s) => {
            let bind2 = match bind.as_ref() {
                Bind::NonRec(v, r) => Bind::NonRec(var(v), Box::new(substitute(r, subst))),
                Bind::Rec(bs) => Bind::Rec(
                    bs.iter()
                        .map(|(v, r)| (var(v), Box::new(substitute(r, subst))))
                        .collect(),
                ),
            };
            Expr::Let(Box::new(bind2), Box::new(substitute(body, subst)), *s)
        }
        Expr::Case(scrut, alts, t, s) => {
            // A default alternative's binder is the scrutinee itself (the
            // worker/wrapper unboxing `case x of { _ x$ww -> … }`); when it was
            // left untyped it takes the scrutinee's type, or `x$ww + x$ww` in
            // the copy has nothing to dispatch on.
            let scrut_ty = match scrut.as_ref() {
                Expr::Var(v, _) => Some(subst.apply(&v.ty)),
                _ => None,
            };
            let alt_binder = |a: &Alt, b: &Var| {
                let typed = var(b);
                match (&a.con, &typed.ty, &scrut_ty) {
                    (crate::AltCon::Default, Ty::Error, Some(st)) if !matches!(st, Ty::Error) => {
                        Var::new(typed.name, typed.id, st.clone())
                    }
                    _ => typed,
                }
            };
            Expr::Case(
                Box::new(substitute(scrut, subst)),
                alts.iter()
                    .map(|a| Alt {
                        con: a.con.clone(),
                        binders: a.binders.iter().map(|b| alt_binder(a, b)).collect(),
                        rhs: substitute(&a.rhs, subst),
                    })
                    .collect(),
                subst.apply(t),
                *s,
            )
        }
        Expr::Type(t, s) => Expr::Type(subst.apply(t), *s),
        Expr::Coercion(c, s) => Expr::Coercion(c.clone(), *s),
    }
}

/// Spread the copy's now-concrete types to what the generic body left
/// untyped: an argument lambda's binders take the parameter types of the
/// function it is passed to (`foldr (\\x acc -> …) 0 xs` at `Integer`), a
/// machine literal in an `Integer`/`Double` parameter position becomes a
/// literal of that type, and an untyped occurrence of a typed binder takes
/// the binder's type. Codegen dispatches arithmetic on these types.
fn propagate(e: &mut Expr, env: &mut FxHashMap<VarId, Ty>) {
    match e {
        Expr::Var(v, _) => {
            if matches!(v.ty, Ty::Error) {
                if let Some(t) = env.get(&v.id) {
                    v.ty = t.clone();
                }
            }
        }
        Expr::App(..) => {
            // The spine: head and arguments, outermost application first.
            let mut params: Vec<Ty> = Vec::new();
            {
                let mut spine: Vec<&Expr> = Vec::new();
                let mut cur: &Expr = e;
                while let Expr::App(f, x, _) = cur {
                    spine.push(x);
                    cur = f;
                }
                if let Expr::Var(h, _) = cur {
                    let mut t = env.get(&h.id).cloned().unwrap_or_else(|| h.ty.clone());
                    for _ in 0..spine.len() {
                        match t {
                            Ty::Fun(a, r) => {
                                params.push(*a);
                                t = *r;
                            }
                            _ => break,
                        }
                    }
                }
            }
            // Walk the spine again, mutably, pairing args with `params`.
            fn walk_args(
                e: &mut Expr,
                params: &[Ty],
                depth: usize,
                env: &mut FxHashMap<VarId, Ty>,
            ) {
                if let Expr::App(f, x, _) = e {
                    walk_args(f, params, depth + 1, env);
                    // `depth` counts from the outermost application, so this
                    // argument is at position `n_args - 1 - depth`.
                    let n_args = depth_of(f) + 1;
                    let pos = n_args - 1;
                    if let Some(p) = params.get(pos) {
                        type_arg(x, p, env);
                    }
                    propagate(x, env);
                } else {
                    propagate(e, env);
                }
            }
            fn depth_of(e: &Expr) -> usize {
                match e {
                    Expr::App(f, _, _) => depth_of(f) + 1,
                    _ => 0,
                }
            }
            walk_args(e, &params, 0, env);
        }
        Expr::Lam(v, b, _) => {
            if !matches!(v.ty, Ty::Error) {
                env.insert(v.id, v.ty.clone());
            }
            propagate(b, env);
        }
        Expr::TyApp(f, _, _)
        | Expr::TyLam(_, f, _)
        | Expr::Lazy(f, _)
        | Expr::Cast(f, _, _)
        | Expr::Tick(_, f, _) => propagate(f, env),
        Expr::Let(bind, body, _) => {
            match bind.as_mut() {
                Bind::NonRec(v, r) => {
                    propagate(r, env);
                    if !matches!(v.ty, Ty::Error) {
                        env.insert(v.id, v.ty.clone());
                    }
                }
                Bind::Rec(bs) => {
                    for (v, _) in bs.iter() {
                        if !matches!(v.ty, Ty::Error) {
                            env.insert(v.id, v.ty.clone());
                        }
                    }
                    for (_, r) in bs.iter_mut() {
                        propagate(r, env);
                    }
                }
            }
            propagate(body, env);
        }
        Expr::Case(scrut, alts, _, _) => {
            propagate(scrut, env);
            for a in alts {
                for b in &a.binders {
                    if !matches!(b.ty, Ty::Error) {
                        env.insert(b.id, b.ty.clone());
                    }
                }
                propagate(&mut a.rhs, env);
            }
        }
        Expr::Lit(..) | Expr::Type(..) | Expr::Coercion(..) => {}
    }
}

/// Give argument `x` the parameter type `p`: an untyped lambda's binders
/// (through nested lambdas), or a machine literal at `Integer`/`Double`.
fn type_arg(x: &mut Expr, p: &Ty, env: &mut FxHashMap<VarId, Ty>) {
    match x {
        Expr::Lam(v, body, _) => {
            if let Ty::Fun(a, r) = p {
                if matches!(v.ty, Ty::Error) && !has_free_tyvar(a) {
                    v.ty = (**a).clone();
                }
                env.insert(v.id, v.ty.clone());
                type_arg(body, r, env);
            }
        }
        Expr::Lit(Literal::Int(n), t, s) => {
            if let Ty::Con(c) = p {
                match c.name.as_str() {
                    "Integer" => {
                        *x = Expr::Lit(Literal::Integer(i128::from(*n)), p.clone(), *s);
                    }
                    "Double" | "Float" => {
                        #[allow(clippy::cast_precision_loss)]
                        let d = *n as f64;
                        *x = Expr::Lit(Literal::Double(d), p.clone(), *s);
                    }
                    _ => {
                        let _ = t;
                    }
                }
            }
        }
        _ => {}
    }
}
