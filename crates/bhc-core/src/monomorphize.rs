//! Monomorphize polymorphic-monad functions at concrete transformer stacks.
//!
//! See `spec/BHC-BRIEF-0004`. A function like
//!
//! ```haskell
//! poly :: Monad m => Int -> StateT Int m Int
//! poly x = do { modify (+ x); s <- get; return s }
//! ```
//!
//! is lowered ONCE from its declared type. Codegen chooses a transformer's
//! runtime representation (closure arity) from that type; a free monad variable
//! `m` yields the stack `[StateT]` and the 2-arg StateT-over-IO closure. When the
//! same function is used at a concrete `m = ExceptT e (StateT s IO)` the correct
//! representation is the 3-arg "stes" closure, so the single emitted body has the
//! wrong arity and the caller's `evalStateT` segfaults.
//!
//! The concrete instantiation is not in the Core (there are no `TyApp`s for
//! ordinary polymorphic calls, and occurrence `Var.ty`s stay polymorphic); it
//! survives only in typeck's span-keyed `resolved_expr_types`. This pass reads
//! that map, and for each top-level binding used at a concrete transformer stack
//! it emits a specialized copy with the monad variable substituted throughout —
//! so codegen re-derives the right stack from the specialized (now concrete)
//! type. The original binding is left in place.
//!
//! The binding may be defined in THIS module or an imported one: the driver
//! transports every module's Core bodies (and newtype/synonym definitions) in a
//! `.bhc` sidecar next to its `.bhi`, and passes the imported bodies (`imported`)
//! and type expansions (`newtypes`) here. So a `PandocMonad m =>` writer defined
//! in one module and run at a concrete `PandocIO` (a newtype over the stes stack)
//! in another specializes in the driver module, transitively through the whole
//! writer chain.

use crate::simplify::expr_util::fresh_var_id;
use crate::{Alt, Bind, CoreModule, Expr, Var, VarId};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Subst, Ty, TyVar};
use rustc_hash::FxHashMap;

/// Specialize polymorphic-monad functions at their concrete transformer stacks.
///
/// `imported` carries the Core bodies of top-level bindings from modules this one
/// imports (transported via `.bhc` sidecars, keyed by binder name); it lets the
/// pass specialize a `PandocMonad m =>` writer defined in another module at the
/// concrete monad the current module pins — the cross-module case. When empty the
/// pass is same-module only.
///
/// Returns the number of specialized bindings created.
pub fn monomorphize_module(
    module: &mut CoreModule,
    resolved: &FxHashMap<Span, Ty>,
    imported: &FxHashMap<String, (Var, Expr)>,
    newtypes: &FxHashMap<bhc_intern::Symbol, (Vec<TyVar>, Ty)>,
) -> usize {
    // Index every module-local top-level binding by VarId (NonRec + Rec members).
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
        imported,
        newtypes,
        memo: FxHashMap::default(),
        cur_monad: Ty::Error,
        new_bindings: Vec::new(),
        resolved,
    };

    // Redirect concrete-instantiation occurrences in the ORIGINAL bodies, using
    // the span-keyed resolved-type map. Cloning specialized bodies (and recursing
    // into them) happens on demand inside `get_or_specialize`.
    let mut bindings = std::mem::take(&mut module.bindings);
    for bind in &mut bindings {
        match bind {
            Bind::NonRec(_, e) => redirect_seeds(e, resolved, &mut ctx),
            Bind::Rec(bs) => {
                for (_, e) in bs {
                    redirect_seeds(e, resolved, &mut ctx);
                }
            }
        }
    }
    let created = ctx.new_bindings.len();
    bindings.append(&mut ctx.new_bindings);
    module.bindings = bindings;
    created
}

/// Give every `VarId` in these bindings a fresh, globally-unique id (consistently:
/// a binder and all its uses stay linked). Used on Core bodies transported from
/// another module via a `.bhc` sidecar, whose ids belong to the source module and
/// would otherwise collide with the importing module's own ids. Names and types
/// are untouched, so name-based references and codegen's name-keyed builtin/extern
/// dispatch are unaffected.
pub fn refresh_var_ids(bindings: &mut [Bind]) {
    let mut map: FxHashMap<VarId, VarId> = FxHashMap::default();
    for b in bindings.iter_mut() {
        refresh_bind(b, &mut map);
    }
}

fn refresh_var(v: &mut Var, map: &mut FxHashMap<VarId, VarId>) {
    v.id = *map.entry(v.id).or_insert_with(fresh_var_id);
}

fn refresh_bind(b: &mut Bind, map: &mut FxHashMap<VarId, VarId>) {
    match b {
        Bind::NonRec(v, e) => {
            refresh_var(v, map);
            refresh_expr(e, map);
        }
        Bind::Rec(bs) => {
            for (v, e) in bs {
                refresh_var(v, map);
                refresh_expr(e, map);
            }
        }
    }
}

fn refresh_expr(e: &mut Expr, map: &mut FxHashMap<VarId, VarId>) {
    match e {
        Expr::Var(v, _) => refresh_var(v, map),
        Expr::App(f, x, _) => {
            refresh_expr(f, map);
            refresh_expr(x, map);
        }
        Expr::TyApp(f, _, _) => refresh_expr(f, map),
        Expr::Lam(v, b, _) => {
            refresh_var(v, map);
            refresh_expr(b, map);
        }
        Expr::TyLam(_, b, _) | Expr::Lazy(b, _) | Expr::Cast(b, _, _) | Expr::Tick(_, b, _) => {
            refresh_expr(b, map);
        }
        Expr::Let(bind, body, _) => {
            refresh_bind(bind, map);
            refresh_expr(body, map);
        }
        Expr::Case(s, alts, _, _) => {
            refresh_expr(s, map);
            for a in alts {
                for bd in &mut a.binders {
                    refresh_var(bd, map);
                }
                refresh_expr(&mut a.rhs, map);
            }
        }
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
    }
}

struct Ctx<'a> {
    /// Module-local top-level bindings, by VarId.
    top: FxHashMap<VarId, (Var, Expr)>,
    /// Imported top-level bodies, by binder name (from `.bhc` sidecars).
    imported: &'a FxHashMap<String, (Var, Expr)>,
    /// Newtype definitions (name -> (params, underlying type)), so a concrete
    /// instantiation written as a newtype over a transformer stack (e.g. pandoc's
    /// `PandocIO`) is unfolded to that stack before transformer detection.
    newtypes: &'a FxHashMap<bhc_intern::Symbol, (Vec<TyVar>, Ty)>,
    /// Memo of specializations, keyed by (binder name, concrete-type string).
    memo: FxHashMap<(String, String), Var>,
    /// The concrete monad of the specialization currently being cloned. Used to
    /// specialize a callee whose occurrence type was erased (`Ty::Error`): every
    /// polymorphic-monad call inside a clone runs in this monad. `Ty::Error` at the
    /// top level (no active specialization).
    cur_monad: Ty,
    /// Specialized bindings produced by this pass.
    new_bindings: Vec<Bind>,
    /// Typeck's span-keyed occurrence types (final substitution). Occurrence
    /// `Var.ty`s in Core can use per-occurrence fresh tyvars that are NOT the
    /// enclosing binding's quantified variable; this map is more consistent, so
    /// the clone walk prefers it (keyed by the occurrence's preserved span).
    resolved: &'a FxHashMap<Span, Ty>,
}

/// Rewrite occurrences in an ORIGINAL body: an occurrence of a module-local
/// polymorphic-monad binding whose `resolved_expr_types[span]` is a concrete
/// transformer stack is redirected to a freshly specialized copy.
fn redirect_seeds(e: &mut Expr, resolved: &FxHashMap<Span, Ty>, ctx: &mut Ctx) {
    match e {
        Expr::Var(v, span) => {
            if let Some(spec) = try_seed(v, *span, resolved, ctx) {
                *e = Expr::Var(spec, *span);
            }
        }
        Expr::App(f, x, _) => {
            redirect_seeds(f, resolved, ctx);
            redirect_seeds(x, resolved, ctx);
        }
        Expr::TyApp(f, _, _)
        | Expr::Lam(_, f, _)
        | Expr::TyLam(_, f, _)
        | Expr::Lazy(f, _)
        | Expr::Cast(f, _, _)
        | Expr::Tick(_, f, _) => redirect_seeds(f, resolved, ctx),
        Expr::Let(bind, body, _) => {
            match bind.as_mut() {
                Bind::NonRec(_, r) => redirect_seeds(r, resolved, ctx),
                Bind::Rec(bs) => {
                    for (_, r) in bs {
                        redirect_seeds(r, resolved, ctx);
                    }
                }
            }
            redirect_seeds(body, resolved, ctx);
        }
        Expr::Case(s, alts, _, _) => {
            redirect_seeds(s, resolved, ctx);
            for a in alts {
                redirect_seeds(&mut a.rhs, resolved, ctx);
            }
        }
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
    }
}

/// If `v` (at occurrence `span`) is a module-local polymorphic-monad binding
/// instantiated at a concrete transformer stack (per `resolved`), return the
/// specialized variable to use in its place.
impl Ctx<'_> {
    /// The source binding for an occurrence: a module-local binding (by VarId) or
    /// an imported one (by binder name, from a `.bhc` sidecar).
    ///
    /// The by-id lookup is validated by name: inside a clone of an IMPORTED body
    /// the occurrence ids belong to the source module and can collide with this
    /// module's own VarIds, so a bare `top.get(id)` may return an unrelated local
    /// binding. When the names disagree we fall through to the imported map.
    fn source_binding(&self, v: &Var) -> Option<(Var, Expr)> {
        if let Some(b) = self.top.get(&v.id) {
            if b.0.name == v.name {
                return Some(b.clone());
            }
        }
        self.imported.get(v.name.as_str()).cloned()
    }
}

/// Expand newtype constructors in `ty` to their underlying types (recursively),
/// so a concrete monad written as a newtype over a transformer stack — e.g.
/// pandoc's `PandocIO = ExceptT PandocError (StateT CommonState IO)`, or the test
/// `AppM` — is seen as that stack. Newtypes are representationally transparent, so
/// substituting the underlying stack into the specialized body is sound.
fn unfold_newtypes(
    ty: &Ty,
    nts: &FxHashMap<bhc_intern::Symbol, (Vec<TyVar>, Ty)>,
    depth: usize,
) -> Ty {
    if depth > 32 || nts.is_empty() {
        return ty.clone();
    }
    // Peel the application spine to find the head constructor and its arguments.
    let mut head = ty;
    let mut args: Vec<&Ty> = Vec::new();
    while let Ty::App(f, x) = head {
        args.push(x);
        head = f;
    }
    args.reverse();
    if let Ty::Con(c) = head {
        if let Some((params, underlying)) = nts.get(&c.name) {
            if args.len() >= params.len() {
                let mut subst = Subst::new();
                for (p, a) in params.iter().zip(&args) {
                    subst.insert(p, unfold_newtypes(a, nts, depth + 1));
                }
                let mut result = subst.apply(underlying);
                for a in &args[params.len()..] {
                    result = Ty::App(
                        Box::new(result),
                        Box::new(unfold_newtypes(a, nts, depth + 1)),
                    );
                }
                return unfold_newtypes(&result, nts, depth + 1);
            }
        }
    }
    match ty {
        Ty::App(f, x) => Ty::App(
            Box::new(unfold_newtypes(f, nts, depth + 1)),
            Box::new(unfold_newtypes(x, nts, depth + 1)),
        ),
        Ty::Fun(a, b) => Ty::Fun(
            Box::new(unfold_newtypes(a, nts, depth + 1)),
            Box::new(unfold_newtypes(b, nts, depth + 1)),
        ),
        Ty::List(a) => Ty::List(Box::new(unfold_newtypes(a, nts, depth + 1))),
        Ty::Tuple(ts) => Ty::Tuple(
            ts.iter()
                .map(|t| unfold_newtypes(t, nts, depth + 1))
                .collect(),
        ),
        Ty::Forall(vs, b) => Ty::Forall(vs.clone(), Box::new(unfold_newtypes(b, nts, depth + 1))),
        other => other.clone(),
    }
}

fn try_seed(v: &Var, span: Span, resolved: &FxHashMap<Span, Ty>, ctx: &mut Ctx) -> Option<Var> {
    let cty = unfold_newtypes(resolved.get(&span)?, ctx.newtypes, 0);
    if has_free_tyvar(&cty) || !mentions_transformer(&cty) {
        return None;
    }
    let (orig_var, orig_body) = ctx.source_binding(v)?;
    let cm = concrete_monad_of(&orig_var.ty, &cty)?;
    get_or_specialize(&orig_var, &orig_body, &cm, ctx)
}

/// The concrete monad a call is at: the image of the callee's single quantified
/// variable when the callee's RESULT type is matched against the concrete
/// occurrence's result. Only the result is matched, not the whole function type —
/// cross-module occurrence types can carry a wrong ARGUMENT type (pandoc's
/// `writeHtmlString'` records `Text` where its first parameter is `WriterState`),
/// which is harmless: the monad variable lives in the result (`… -> m a` or
/// `… -> StateT s m a`), and matching results extracts it for both shapes.
fn concrete_monad_of(binder_ty: &Ty, conc_ty: &Ty) -> Option<Ty> {
    let mut binder_vars = Vec::new();
    collect_tyvars(binder_ty, &mut binder_vars);
    if binder_vars.len() != 1 {
        return None;
    }
    let mut m = Subst::new();
    if !match_ty(ultimate_result(binder_ty), ultimate_result(conc_ty), &mut m) {
        return None;
    }
    m.get(&binder_vars[0])
        .cloned()
        .filter(|c| !has_free_tyvar(c))
}

/// Get or create the specialization of a top-level binding (local or imported) at
/// `concrete_monad` — the monad its single quantified variable maps to.
///
/// Only single-monad bindings are specialized: the binder type must have EXACTLY
/// ONE free type variable (the monad `m`; everything else already concrete). Every
/// free type variable in the body then denotes that same `m` (regardless of the
/// per-occurrence fresh ids Core assigns), so all map to `concrete_monad`. A
/// binding with more than one free variable is left untouched — mapping its
/// non-monad variables would be unsound.
fn get_or_specialize(
    orig_var: &Var,
    orig_body: &Expr,
    concrete_monad: &Ty,
    ctx: &mut Ctx,
) -> Option<Var> {
    let mut binder_vars = Vec::new();
    collect_tyvars(&orig_var.ty, &mut binder_vars);
    if binder_vars.len() != 1 {
        return None;
    }
    let mb = binder_vars[0].clone();
    if has_free_tyvar(concrete_monad) || !mentions_transformer(concrete_monad) {
        return None;
    }
    let concrete_monad = concrete_monad.clone();

    let key = (orig_var.name.to_string(), format!("{concrete_monad:?}"));
    if let Some(spec) = ctx.memo.get(&key) {
        return Some(spec.clone());
    }

    // Map the binder variable AND every free variable that appears in the body's
    // occurrence types (Core `Var.ty` and typeck's `resolved`) to the concrete
    // monad. For a single-monad binding these are all the same `m`.
    let mut subst = Subst::new();
    subst.insert(&mb, concrete_monad.clone());
    let mut body_vars = Vec::new();
    collect_body_tyvars(orig_body, ctx.resolved, &mut body_vars);
    for tv in body_vars {
        subst.insert(&tv, concrete_monad.clone());
    }

    let new_ty = subst.apply(&orig_var.ty);
    let new_id = fresh_var_id();
    let new_name = Symbol::intern(&format!(
        "{}$$mono{}",
        orig_var.name.as_str(),
        new_id.index()
    ));
    let spec_var = Var::new(new_name, new_id, new_ty);
    // Memoize BEFORE recursing so a (mutually) recursive body resolves to this
    // same specialization instead of looping.
    ctx.memo.insert(key, spec_var.clone());
    let saved_monad = std::mem::replace(&mut ctx.cur_monad, concrete_monad.clone());
    let spec_body = specialize_body(orig_body, &subst, ctx);
    ctx.cur_monad = saved_monad;
    ctx.new_bindings
        .push(Bind::NonRec(spec_var.clone(), Box::new(spec_body)));
    Some(spec_var)
}

/// Collect every free type variable that appears in the body's occurrence types,
/// preferring typeck's span-keyed `resolved` type over the Core `Var.ty` (the
/// former is internally consistent; the latter can use a different fresh id for
/// the same variable). Also covers types embedded in `Lit`/`Case`/binders.
fn collect_body_tyvars(e: &Expr, resolved: &FxHashMap<Span, Ty>, out: &mut Vec<TyVar>) {
    match e {
        Expr::Var(v, span) => {
            collect_tyvars(resolved.get(span).unwrap_or(&v.ty), out);
        }
        Expr::Lit(_, ty, _) | Expr::Type(ty, _) => collect_tyvars(ty, out),
        Expr::App(f, x, _) => {
            collect_body_tyvars(f, resolved, out);
            collect_body_tyvars(x, resolved, out);
        }
        Expr::TyApp(f, ty, _) => {
            collect_body_tyvars(f, resolved, out);
            collect_tyvars(ty, out);
        }
        Expr::Lam(v, b, _) => {
            collect_tyvars(&v.ty, out);
            collect_body_tyvars(b, resolved, out);
        }
        Expr::TyLam(_, b, _) | Expr::Lazy(b, _) | Expr::Cast(b, _, _) | Expr::Tick(_, b, _) => {
            collect_body_tyvars(b, resolved, out);
        }
        Expr::Let(bind, body, _) => {
            match bind.as_ref() {
                Bind::NonRec(v, r) => {
                    collect_tyvars(&v.ty, out);
                    collect_body_tyvars(r, resolved, out);
                }
                Bind::Rec(bs) => {
                    for (v, r) in bs {
                        collect_tyvars(&v.ty, out);
                        collect_body_tyvars(r, resolved, out);
                    }
                }
            }
            collect_body_tyvars(body, resolved, out);
        }
        Expr::Case(s, alts, ty, _) => {
            collect_body_tyvars(s, resolved, out);
            collect_tyvars(ty, out);
            for a in alts {
                for b in &a.binders {
                    collect_tyvars(&b.ty, out);
                }
                collect_body_tyvars(&a.rhs, resolved, out);
            }
        }
        Expr::Coercion(_, _) => {}
    }
}

/// Deep-clone `e`, applying `subst` to every embedded type, and redirecting inner
/// occurrences of other module-local polymorphic-monad bindings (whose occurrence
/// type is now ground after substitution) to their own specializations.
fn specialize_body(e: &Expr, subst: &Subst, ctx: &mut Ctx) -> Expr {
    match e {
        Expr::Var(v, span) => {
            // Occurrence `Var.ty`s can use per-occurrence fresh tyvars unrelated
            // to the enclosing binding's quantified variable, so prefer typeck's
            // span-keyed occurrence type (which is consistent) as the base to
            // substitute into; fall back to the Core `Var.ty` (the only source for
            // an imported clone body, whose spans are not in this module's map).
            let new_ty = {
                let base_ty = ctx.resolved.get(span).unwrap_or(&v.ty);
                unfold_newtypes(&subst.apply(base_ty), ctx.newtypes, 0)
            };
            // A usable occurrence type that is a ground transformer stack: specialize
            // the callee at the monad that type names.
            if !has_free_tyvar(&new_ty) && mentions_transformer(&new_ty) {
                if let Some((ov, ob)) = ctx.source_binding(v) {
                    if has_free_tyvar(&ov.ty) {
                        if let Some(cm) = concrete_monad_of(&ov.ty, &new_ty) {
                            if let Some(spec) = get_or_specialize(&ov, &ob, &cm, ctx) {
                                return Expr::Var(spec, *span);
                            }
                        }
                    }
                }
            } else if matches!(ctx.resolved.get(span).unwrap_or(&v.ty), Ty::Error)
                && mentions_transformer(&ctx.cur_monad)
            {
                // The occurrence type was ERASED (pandoc's `lift $ setupTranslations
                // meta` records `Ty::Error`), so there is no transformer to read from
                // it. But every polymorphic-monad call inside a clone specialized at
                // `cur_monad` runs in that same monad (the writer's inner monad IS it,
                // even under `lift`), so specialize the callee at `cur_monad`.
                let cur = ctx.cur_monad.clone();
                if let Some((ov, ob)) = ctx.source_binding(v) {
                    if has_free_tyvar(&ov.ty) {
                        if let Some(spec) = get_or_specialize(&ov, &ob, &cur, ctx) {
                            return Expr::Var(spec, *span);
                        }
                    }
                }
            }
            Expr::Var(Var::new(v.name, v.id, new_ty), *span)
        }
        Expr::Lit(l, ty, span) => Expr::Lit(l.clone(), subst.apply(ty), *span),
        Expr::App(f, x, span) => {
            // A `Monad m` do-block lowers `>>=`/`>>` as a dictionary method
            // selection `$sel_N $dMonad`; the transformer-stack Monad dictionary
            // is a null placeholder, so the compiled body reads a null slot and
            // bails. In the specialized (concrete-monad) copy we rewrite that
            // selection to the plain builtin operator, which codegen then routes
            // by the concrete ambient transformer stack (e.g. to `stes_bind`) —
            // exactly as a hand-written concrete do-block is compiled. Every
            // `specialize_body` runs under a concrete transformer monad (that is
            // the only thing `get_or_specialize` fires for), so this is sound.
            if let (Expr::Var(sel, _), Expr::Var(d, _)) = (f.as_ref(), x.as_ref()) {
                if d.name.as_str().starts_with("$dMonad") {
                    if let Some(op) = monad_sel_builtin(sel.name.as_str()) {
                        return Expr::Var(
                            Var::new(Symbol::intern(op), fresh_var_id(), Ty::Error),
                            *span,
                        );
                    }
                }
            }
            Expr::App(
                Box::new(specialize_body(f, subst, ctx)),
                Box::new(specialize_body(x, subst, ctx)),
                *span,
            )
        }
        Expr::TyApp(f, ty, span) => Expr::TyApp(
            Box::new(specialize_body(f, subst, ctx)),
            subst.apply(ty),
            *span,
        ),
        Expr::Lam(v, b, span) => Expr::Lam(
            Var::new(v.name, v.id, subst.apply(&v.ty)),
            Box::new(specialize_body(b, subst, ctx)),
            *span,
        ),
        Expr::TyLam(tv, b, span) => {
            Expr::TyLam(tv.clone(), Box::new(specialize_body(b, subst, ctx)), *span)
        }
        Expr::Let(bind, body, span) => Expr::Let(
            Box::new(specialize_bind(bind, subst, ctx)),
            Box::new(specialize_body(body, subst, ctx)),
            *span,
        ),
        Expr::Case(s, alts, ty, span) => Expr::Case(
            Box::new(specialize_body(s, subst, ctx)),
            alts.iter()
                .map(|a| Alt {
                    con: a.con.clone(),
                    binders: a
                        .binders
                        .iter()
                        .map(|b| Var::new(b.name, b.id, subst.apply(&b.ty)))
                        .collect(),
                    rhs: specialize_body(&a.rhs, subst, ctx),
                })
                .collect(),
            subst.apply(ty),
            *span,
        ),
        Expr::Lazy(b, span) => Expr::Lazy(Box::new(specialize_body(b, subst, ctx)), *span),
        Expr::Cast(b, c, span) => {
            Expr::Cast(Box::new(specialize_body(b, subst, ctx)), c.clone(), *span)
        }
        Expr::Tick(t, b, span) => {
            Expr::Tick(t.clone(), Box::new(specialize_body(b, subst, ctx)), *span)
        }
        Expr::Type(ty, span) => Expr::Type(subst.apply(ty), *span),
        Expr::Coercion(c, span) => Expr::Coercion(c.clone(), *span),
    }
}

fn specialize_bind(bind: &Bind, subst: &Subst, ctx: &mut Ctx) -> Bind {
    match bind {
        Bind::NonRec(v, e) => Bind::NonRec(
            Var::new(v.name, v.id, subst.apply(&v.ty)),
            Box::new(specialize_body(e, subst, ctx)),
        ),
        Bind::Rec(bs) => Bind::Rec(
            bs.iter()
                .map(|(v, e)| {
                    (
                        Var::new(v.name, v.id, subst.apply(&v.ty)),
                        Box::new(specialize_body(e, subst, ctx)),
                    )
                })
                .collect(),
        ),
    }
}

/// One-sided structural match: bind `pat`'s type variables so that it becomes
/// `conc`. Fails on any inconsistency.
fn match_ty(pat: &Ty, conc: &Ty, subst: &mut Subst) -> bool {
    match (pat, conc) {
        (Ty::Var(v), _) => match subst.get(v) {
            Some(existing) => existing == conc,
            None => {
                subst.insert(v, conc.clone());
                true
            }
        },
        (Ty::Con(a), Ty::Con(b)) => a.name == b.name,
        (Ty::Prim(a), Ty::Prim(b)) => a == b,
        (Ty::App(a1, a2), Ty::App(b1, b2)) | (Ty::Fun(a1, a2), Ty::Fun(b1, b2)) => {
            match_ty(a1, b1, subst) && match_ty(a2, b2, subst)
        }
        (Ty::List(a), Ty::List(b)) => match_ty(a, b, subst),
        (Ty::Tuple(a), Ty::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| match_ty(x, y, subst))
        }
        _ => pat == conc,
    }
}

/// Map a `Monad` dictionary field selector to its builtin operator. The `Monad`
/// dictionary lays out `[Applicative superclass, (>>=), (>>)]`, so `$sel_1`
/// selects `>>=` and `$sel_2` selects `>>`.
fn monad_sel_builtin(sel: &str) -> Option<&'static str> {
    match sel {
        "$sel_1" => Some(">>="),
        "$sel_2" => Some(">>"),
        _ => None,
    }
}

/// The ultimate result of a function type (all leading `->` arrows stripped).
fn ultimate_result(ty: &Ty) -> &Ty {
    let mut r = ty;
    while let Ty::Fun(_, b) = r {
        r = b;
    }
    r
}

fn has_free_tyvar(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => true,
        Ty::App(a, b) | Ty::Fun(a, b) => has_free_tyvar(a) || has_free_tyvar(b),
        Ty::List(a) => has_free_tyvar(a),
        Ty::Tuple(ts) => ts.iter().any(has_free_tyvar),
        Ty::Forall(_, b) => has_free_tyvar(b),
        _ => false,
    }
}

fn mentions_transformer(ty: &Ty) -> bool {
    match ty {
        Ty::Con(c) => matches!(
            c.name.as_str(),
            "StateT" | "ExceptT" | "ReaderT" | "WriterT"
        ),
        Ty::App(a, b) | Ty::Fun(a, b) => mentions_transformer(a) || mentions_transformer(b),
        Ty::List(a) => mentions_transformer(a),
        Ty::Tuple(ts) => ts.iter().any(mentions_transformer),
        Ty::Forall(_, b) => mentions_transformer(b),
        _ => false,
    }
}

fn collect_tyvars(ty: &Ty, out: &mut Vec<TyVar>) {
    match ty {
        Ty::Var(v) => {
            if !out.contains(v) {
                out.push(v.clone());
            }
        }
        Ty::App(a, b) | Ty::Fun(a, b) => {
            collect_tyvars(a, out);
            collect_tyvars(b, out);
        }
        Ty::List(a) => collect_tyvars(a, out),
        Ty::Tuple(ts) => ts.iter().for_each(|t| collect_tyvars(t, out)),
        Ty::Forall(_, b) => collect_tyvars(b, out),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_types::{Kind, TyCon};

    fn con(name: &str) -> Ty {
        Ty::Con(TyCon::new(Symbol::intern(name), Kind::Star))
    }
    fn var(id: u32) -> Ty {
        Ty::Var(TyVar::new(id, Kind::Star))
    }
    fn app(f: Ty, x: Ty) -> Ty {
        Ty::App(Box::new(f), Box::new(x))
    }
    fn fun(a: Ty, b: Ty) -> Ty {
        Ty::Fun(Box::new(a), Box::new(b))
    }
    /// `StateT Int m Int`
    fn state_t(m: Ty) -> Ty {
        app(app(app(con("StateT"), con("Int")), m), con("Int"))
    }
    /// `ExceptT String (StateT Int IO)` — the concrete inner monad.
    fn stes_inner() -> Ty {
        app(
            app(con("ExceptT"), con("String")),
            app(app(con("StateT"), con("Int")), con("IO")),
        )
    }

    #[test]
    fn match_binds_the_monad_variable() {
        // `Int -> StateT Int m Int`  vs  `Int -> StateT Int (ExceptT ..) Int`
        let poly = fun(con("Int"), state_t(var(55)));
        let conc = fun(con("Int"), state_t(stes_inner()));
        let mut s = Subst::new();
        assert!(match_ty(&poly, &conc, &mut s));
        assert_eq!(s.get(&TyVar::new(55, Kind::Star)), Some(&stes_inner()));
    }

    #[test]
    fn match_fails_on_mismatch() {
        let poly = fun(con("Int"), state_t(var(55)));
        let conc = fun(con("Bool"), state_t(stes_inner())); // Int vs Bool
        let mut s = Subst::new();
        assert!(!match_ty(&poly, &conc, &mut s));
    }

    #[test]
    fn free_and_transformer_predicates() {
        assert!(has_free_tyvar(&state_t(var(55))));
        assert!(!has_free_tyvar(&state_t(stes_inner())));
        assert!(mentions_transformer(&state_t(stes_inner())));
        assert!(!mentions_transformer(&fun(con("Int"), con("Int"))));
    }

    #[test]
    fn one_free_tyvar_for_single_monad_binding() {
        let mut vs = Vec::new();
        collect_tyvars(&fun(con("Int"), state_t(var(55))), &mut vs);
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], TyVar::new(55, Kind::Star));
    }

    #[test]
    fn monad_selectors_map_to_operators() {
        assert_eq!(monad_sel_builtin("$sel_1"), Some(">>="));
        assert_eq!(monad_sel_builtin("$sel_2"), Some(">>"));
        assert_eq!(monad_sel_builtin("$sel_0"), None);
        assert_eq!(monad_sel_builtin("modify"), None);
    }

    #[test]
    fn newtype_unfolds_to_underlying_stack() {
        // `newtype AppM a = AppM (ExceptT String (StateT Int IO) a)`
        let a = TyVar::new(1, Kind::Star);
        let underlying = app(
            app(con("ExceptT"), con("String")),
            app(
                app(app(con("StateT"), con("Int")), con("IO")),
                Ty::Var(a.clone()),
            ),
        );
        let mut nts: FxHashMap<Symbol, (Vec<TyVar>, Ty)> = FxHashMap::default();
        nts.insert(Symbol::intern("AppM"), (vec![a], underlying));
        // `AppM Int` unfolds to `ExceptT String (StateT Int IO) Int`.
        let unfolded = unfold_newtypes(&app(con("AppM"), con("Int")), &nts, 0);
        assert!(mentions_transformer(&unfolded));
        assert!(!has_free_tyvar(&unfolded));
        let expected = app(
            app(con("ExceptT"), con("String")),
            app(app(app(con("StateT"), con("Int")), con("IO")), con("Int")),
        );
        assert_eq!(unfolded, expected);
        // A plain non-newtype type is unchanged.
        assert_eq!(unfold_newtypes(&con("Int"), &nts, 0), con("Int"));
    }
}
