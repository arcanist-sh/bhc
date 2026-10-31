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
use rustc_hash::{FxHashMap, FxHashSet};

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
        spec_counts: FxHashMap::default(),
        in_imported: false,
        local_binders: FxHashSet::default(),
        cur_monad: Ty::Error,
        monad_dicts: FxHashSet::default(),
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
    if let Ok(filter) = std::env::var("BHC_DUMP_MONO") {
        for b in ctx.new_bindings.iter().chain(bindings.iter()) {
            if let Bind::NonRec(v, e) = b {
                if v.name.as_str().contains(filter.as_str()) {
                    eprintln!(
                        "=== MONO BIND {} :: {:?} ===\n{:#?}",
                        v.name.as_str(),
                        v.ty,
                        e
                    );
                }
            }
        }
    }
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
    /// Specializations made per binder name — bounds polymorphic recursion
    /// (see `MAX_SPECS_PER_BINDING`).
    spec_counts: FxHashMap<String, usize>,
    /// Whether the body being specialized is IMPORTED. Its spans belong to
    /// another file, so this module's span-keyed `resolved` types must not be
    /// read for it — a coincidental span match handed an imported body's
    /// occurrences types from unrelated local expressions.
    in_imported: bool,
    /// Variables BOUND inside the body being specialized (lambda, let and case
    /// binders). Imported bindings are found by NAME, so a local `\w -> …`
    /// in an imported body was taken for a call to an imported top-level `w`
    /// and specialized — recursively, at an ever deeper monad.
    local_binders: FxHashSet<VarId>,
    /// The concrete monad of the specialization currently being cloned. Used to
    /// specialize a callee whose occurrence type was erased (`Ty::Error`): every
    /// polymorphic-monad call inside a clone runs in this monad. `Ty::Error` at the
    /// top level (no active specialization).
    cur_monad: Ty,
    /// Dictionary variables of the clone being built that denote ITS monad:
    /// the binding's own lambda-bound dictionary parameters and superclass
    /// dictionaries selected out of them. Only a selection from one of these
    /// may be rewritten to a builtin operator — a Monad dictionary the body
    /// CONSTRUCTED for some other monad (a parser's `ParsecT` instance inside
    /// `myRead :: PandocMonad m => …`) is named `$dMonad_NNN` just the same,
    /// and routing its `>>` through the specialized stack ran the parser as a
    /// PandocIO action.
    monad_dicts: FxHashSet<VarId>,
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

    /// Whether `v`'s source binding is module-local (else imported).
    fn is_local(&self, v: &Var) -> bool {
        self.top.get(&v.id).is_some_and(|b| b.0.name == v.name)
    }

    /// The span-keyed occurrence type, when the body being specialized is
    /// this module's own.
    fn occ_ty(&self, span: &Span) -> Option<&Ty> {
        if self.in_imported {
            None
        } else {
            self.resolved.get(span)
        }
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
    // `BHC_DBG_MONO=<substring>` traces why an occurrence was or was not seeded.
    let dbg = std::env::var("BHC_DBG_MONO")
        .ok()
        .filter(|f| v.name.as_str().contains(f.as_str()));
    let trace = |what: &str| {
        if dbg.is_some() {
            eprintln!("MONO seed {}: {what}", v.name.as_str());
        }
    };
    // No span-keyed type: fall back to a ground type the reference itself
    // carries — hir-to-core instantiates an instance-method alias's body
    // (`fileExists = IO.fileExists`) at the instance, where typeck records
    // nothing.
    let carried = (!has_free_tyvar(&v.ty) && !matches!(v.ty, Ty::Error)).then_some(&v.ty);
    let Some(rty) = resolved.get(&span).or(carried) else {
        trace("no resolved type at span");
        return None;
    };
    let cty = unfold_newtypes(rty, ctx.newtypes, 0);
    // Only the MONAD's image must be ground (`instantiation_of` checks it): a
    // value type typeck left open (`getsCommonState stVerbosity`'s `a`) does
    // not affect the transformer stack the clone is compiled for.
    if !mentions_transformer(&cty) {
        trace(&format!("no transformer in type: {cty:?}"));
        return None;
    }
    let Some((orig_var, orig_body)) = ctx.source_binding(v) else {
        trace("no source binding");
        return None;
    };
    let Some(inst) = instantiation_of(&orig_var.ty, &cty) else {
        trace(&format!("no instantiation of {:?}", orig_var.ty));
        return None;
    };
    let r = get_or_specialize(&orig_var, &orig_body, &inst, ctx);
    trace(if r.is_some() {
        "specialized"
    } else {
        "get_or_specialize refused"
    });
    r
}

/// The images of a binding's quantified variables at one concrete occurrence:
/// the monad variable's (which decides the clone's transformer stack), and any
/// other variable's that the occurrence fixes to a ground type.
struct Instantiation {
    monad: Ty,
    others: Vec<(TyVar, Ty)>,
}

impl Instantiation {
    fn at_monad(monad: Ty) -> Self {
        Self {
            monad,
            others: Vec::new(),
        }
    }
}

/// The variable that plays the MONAD in a binding's type. With one quantified
/// variable it is that variable (every free variable of such a body denotes
/// it). With several, exactly one may sit in monad position — heading an
/// application (`m a`) or as the argument just before a constructor's final
/// one (`m` in `StateT s m a` and parsec's `ParsecT s u m a`); the final
/// argument is a value type (`a` in `Doc a`, `MarkupM a`). Anything else is
/// not specialized.
fn monad_var_of(binder_ty: &Ty, binder_vars: &[TyVar]) -> Option<TyVar> {
    if binder_vars.len() == 1 {
        return Some(binder_vars[0].clone());
    }
    let mut monadic = binder_vars.iter().filter(|v| var_is_monadic(binder_ty, v));
    let first = monadic.next()?.clone();
    monadic.next().is_none().then_some(first)
}

/// Whether `v` is in monad position in `ty` (see `monad_var_of`).
fn var_is_monadic(ty: &Ty, v: &TyVar) -> bool {
    match ty {
        Ty::App(..) => {
            let mut head = ty;
            let mut args: Vec<&Ty> = Vec::new();
            while let Ty::App(f, x) = head {
                args.push(x);
                head = f;
            }
            // `args` is last-first: `args[0]` is the final argument.
            let heads_it = matches!(head, Ty::Var(h) if h == v);
            // The monad sits just before the final (value) argument:
            // `StateT s m a`, `ParsecT s u m a` — not `s` or `u`.
            let monad_slot =
                matches!(head, Ty::Con(_)) && matches!(args.get(1), Some(Ty::Var(x)) if x == v);
            heads_it || monad_slot || args.iter().any(|a| var_is_monadic(a, v))
        }
        Ty::Fun(a, b) => var_is_monadic(a, v) || var_is_monadic(b, v),
        Ty::List(a) => var_is_monadic(a, v),
        Ty::Tuple(ts) => ts.iter().any(|t| var_is_monadic(t, v)),
        Ty::Forall(_, b) => var_is_monadic(b, v),
        _ => false,
    }
}

/// The instantiation a call is at: the images of the callee's quantified
/// variables when the callee's RESULT type is matched against the concrete
/// occurrence's result. Only the result is matched, not the whole function type —
/// cross-module occurrence types can carry a wrong ARGUMENT type (pandoc's
/// `writeHtmlString'` records `Text` where its first parameter is `WriterState`),
/// which is harmless: the monad variable lives in the result (`… -> m a` or
/// `… -> StateT s m a`), and matching results extracts it for both shapes.
fn instantiation_of(binder_ty: &Ty, conc_ty: &Ty) -> Option<Instantiation> {
    let mut binder_vars = Vec::new();
    collect_tyvars(binder_ty, &mut binder_vars);
    let mv = monad_var_of(binder_ty, &binder_vars)?;
    let mut m = Subst::new();
    let (mut b, mut c) = (ultimate_result(binder_ty), ultimate_result(conc_ty));
    // A dictionary slot shares its use site's span, so the recorded type can
    // be the USE's (`getsCommonState f :: StateT Int PandocIO a`) rather than
    // the slot's (`getCommonState :: t m CommonState`). Their monads agree;
    // only the final value arguments differ — so match with those stripped.
    while !match_ty(b, c, &mut m) {
        m = Subst::new();
        match (b, c) {
            (Ty::App(bf, _), Ty::App(cf, _)) => {
                b = bf;
                c = cf;
            }
            _ => return None,
        }
    }
    let monad = m.get(&mv).cloned().filter(|c| !has_free_tyvar(c))?;
    let others = binder_vars
        .iter()
        .filter(|v| **v != mv)
        .filter_map(|v| {
            m.get(v)
                .filter(|t| !has_free_tyvar(t))
                .map(|t| (v.clone(), t.clone()))
        })
        .collect();
    Some(Instantiation { monad, others })
}

/// Deepest transformer nesting a specialization is made at.
const MAX_TRANSFORMER_DEPTH: usize = 4;

/// Most specializations made of any one binding.
const MAX_SPECS_PER_BINDING: usize = 16;

/// How many transformer constructors are nested in `ty`
/// (`StateT s (ExceptT e (StateT c IO))` is 3).
fn transformer_depth(ty: &Ty) -> usize {
    let mut head = ty;
    let mut args = Vec::new();
    while let Ty::App(f, x) = head {
        args.push(x.as_ref());
        head = f.as_ref();
    }
    let own = usize::from(matches!(head, Ty::Con(c) if is_transformer_name(c.name.as_str())));
    own + args.into_iter().map(transformer_depth).max().unwrap_or(0)
}

/// A transformer constructor applied to MORE than its three arguments
/// (`ExceptT e (StateT s IO) v CommonState`): an ill-kinded type, produced
/// when a binder's `t m a` (both `t` and `m` variables) has only its head `t`
/// mapped to a concrete monad. Specializing at such a type cloned PandocMonad's
/// generic `(t m)` instance methods at nonsense instantiations.
fn over_applied(ty: &Ty) -> bool {
    let mut head = ty;
    let mut n = 0usize;
    while let Ty::App(f, _) = head {
        n += 1;
        head = f;
    }
    let own = matches!(head, Ty::Con(c) if is_transformer_name(c.name.as_str())) && n > 3;
    own || match ty {
        Ty::App(f, x) | Ty::Fun(f, x) => over_applied(f) || over_applied(x),
        Ty::List(a) => over_applied(a),
        Ty::Tuple(ts) => ts.iter().any(over_applied),
        _ => false,
    }
}

fn is_transformer_name(name: &str) -> bool {
    // Library transformers (closure IO model, spec/BHC-BRIEF-0005) are passed
    // dictionaries like any user monad; codegen needs no specialization.
    !bhc_types::io_closure_mode() && matches!(name, "StateT" | "ExceptT" | "ReaderT" | "WriterT")
}

fn get_or_specialize(
    orig_var: &Var,
    orig_body: &Expr,
    inst: &Instantiation,
    ctx: &mut Ctx,
) -> Option<Var> {
    let mut binder_vars = Vec::new();
    collect_tyvars(&orig_var.ty, &mut binder_vars);
    let mb = monad_var_of(&orig_var.ty, &binder_vars)?;
    let concrete_monad = inst.monad.clone();
    if has_free_tyvar(&concrete_monad) || !mentions_transformer(&concrete_monad) {
        return None;
    }

    let others_key: Vec<String> = inst
        .others
        .iter()
        .map(|(v, t)| format!("{}={t:?}", v.id))
        .collect();
    let key = (
        orig_var.name.to_string(),
        format!("{concrete_monad:?}|{}", others_key.join(",")),
    );
    if let Some(spec) = ctx.memo.get(&key) {
        return Some(spec.clone());
    }

    // Map the monad variable AND every free variable that appears in the body's
    // occurrence types (Core `Var.ty` and typeck's `resolved`) to the concrete
    // monad: Core's per-occurrence fresh variables cannot be traced back to the
    // binder's, and for a single-variable binding they all ARE `m`. The other
    // binder variables an occurrence fixed get their own images; their body
    // occurrences fall under the monad mapping, which only steers codegen's
    // transformer-stack decisions (a value type never names a transformer).
    //
    // Only variables in MONAD position (the head of an application, or a
    // transformer's inner-monad argument) are mapped. A value-position one
    // made a record accessor `CommonState -> v` into `CommonState -> <monad>`
    // and `mempty`'s `v` into the monad, and cloned both as monadic code — the
    // clone of a `PandocMonad m =>` helper then called those and crashed.
    let imported_body = !ctx.is_local(orig_var);
    let saved_imported = std::mem::replace(&mut ctx.in_imported, imported_body);
    let mut subst = Subst::new();
    let mut body_vars = Vec::new();
    let no_resolved = FxHashMap::default();
    let body_resolved = if ctx.in_imported {
        &no_resolved
    } else {
        ctx.resolved
    };
    collect_body_tyvars(orig_body, body_resolved, &mut body_vars);
    for tv in body_vars {
        subst.insert(&tv, concrete_monad.clone());
    }
    let mut binder_monad_vars = Vec::new();
    collect_monad_tyvars(&orig_var.ty, &mut binder_monad_vars);
    for v in &binder_monad_vars {
        if *v != mb {
            subst.insert(v, concrete_monad.clone());
        }
    }
    for (v, t) in &inst.others {
        subst.insert(v, t.clone());
    }
    subst.insert(&mb, concrete_monad.clone());

    let new_ty = subst.apply(&orig_var.ty);
    if over_applied(&new_ty) {
        ctx.in_imported = saved_imported;
        return None;
    }
    let new_id = fresh_var_id();
    let new_name = Symbol::intern(&format!(
        "{}$$mono{}",
        orig_var.name.as_str(),
        new_id.index()
    ));
    let spec_var = Var::new(new_name, new_id, new_ty);
    // Polymorphic recursion (pandoc's `toJiraBlocks` recurses at a deeper
    // transformer stack each round) would specialize without end — thousands
    // of ever-larger clones until the compiler was killed. Beyond these bounds
    // the generic, dictionary-passing original is used.
    if transformer_depth(&concrete_monad) > MAX_TRANSFORMER_DEPTH {
        ctx.in_imported = saved_imported;
        return None;
    }
    let count = ctx.spec_counts.entry(key.0.clone()).or_insert(0);
    if *count >= MAX_SPECS_PER_BINDING {
        ctx.in_imported = saved_imported;
        return None;
    }
    *count += 1;
    if std::env::var_os("BHC_DBG_MONO_SPEC").is_some() {
        eprintln!("MONOSPEC {} #{} {}", key.0, count, key.1);
    }
    // Memoize BEFORE recursing so a (mutually) recursive body resolves to this
    // same specialization instead of looping.
    ctx.memo.insert(key, spec_var.clone());
    let saved_monad = std::mem::replace(&mut ctx.cur_monad, concrete_monad.clone());
    let saved_dicts = std::mem::replace(&mut ctx.monad_dicts, own_monad_dicts(orig_body));
    let mut binders = FxHashSet::default();
    collect_binders(orig_body, &mut binders);
    let saved_binders = std::mem::replace(&mut ctx.local_binders, binders);
    let spec_body = specialize_body(orig_body, &subst, ctx);
    ctx.local_binders = saved_binders;
    ctx.cur_monad = saved_monad;
    ctx.monad_dicts = saved_dicts;
    ctx.in_imported = saved_imported;
    ctx.new_bindings
        .push(Bind::NonRec(spec_var.clone(), Box::new(spec_body)));
    Some(spec_var)
}

/// The dictionary variables of a binding body that denote the binding's OWN
/// constrained monad: its leading lambda-bound dictionary parameters, and — to
/// a fixpoint — `let`-bound dictionaries selected out of those
/// (`$superMonad = $sel_k $dPandocMonad`). See `Ctx::monad_dicts`.
fn own_monad_dicts(body: &Expr) -> FxHashSet<VarId> {
    fn derived(rhs: &Expr, set: &FxHashSet<VarId>) -> bool {
        match rhs {
            Expr::Var(v, _) => set.contains(&v.id),
            Expr::App(f, inner, _) => {
                matches!(f.as_ref(), Expr::Var(sel, _) if sel.name.as_str().starts_with("$sel_"))
                    && derived(inner, set)
            }
            _ => false,
        }
    }
    fn walk(e: &Expr, set: &mut FxHashSet<VarId>) {
        match e {
            Expr::Let(bind, body, _) => {
                match bind.as_ref() {
                    Bind::NonRec(x, rhs) => {
                        if dict_class_name(x.name.as_str()).is_some() && derived(rhs, set) {
                            set.insert(x.id);
                        }
                        walk(rhs, set);
                    }
                    Bind::Rec(bs) => {
                        for (_, rhs) in bs {
                            walk(rhs, set);
                        }
                    }
                }
                walk(body, set);
            }
            Expr::App(f, x, _) => {
                walk(f, set);
                walk(x, set);
            }
            Expr::TyApp(f, _, _) | Expr::Lam(_, f, _) | Expr::TyLam(_, f, _) => walk(f, set),
            Expr::Lazy(b, _) | Expr::Cast(b, _, _) | Expr::Tick(_, b, _) => walk(b, set),
            Expr::Case(scrut, alts, _, _) => {
                walk(scrut, set);
                for alt in alts {
                    walk(&alt.rhs, set);
                }
            }
            _ => {}
        }
    }
    let mut set = FxHashSet::default();
    let mut e = body;
    while let Expr::Lam(v, b, _) = e {
        if dict_class_name(v.name.as_str()).is_some() {
            set.insert(v.id);
        }
        e = b;
    }
    loop {
        let before = set.len();
        walk(body, &mut set);
        if set.len() == before {
            break;
        }
    }
    set
}

/// Collect every free type variable that appears in the body's occurrence types,
/// preferring typeck's span-keyed `resolved` type over the Core `Var.ty` (the
/// former is internally consistent; the latter can use a different fresh id for
/// the same variable). Also covers types embedded in `Lit`/`Case`/binders.
/// Every variable bound inside `e` (lambda, let and case-alternative binders).
fn collect_binders(e: &Expr, out: &mut FxHashSet<VarId>) {
    match e {
        Expr::Lam(v, b, _) => {
            out.insert(v.id);
            collect_binders(b, out);
        }
        Expr::Let(bind, body, _) => {
            match bind.as_ref() {
                Bind::NonRec(v, r) => {
                    out.insert(v.id);
                    collect_binders(r, out);
                }
                Bind::Rec(bs) => {
                    for (v, r) in bs {
                        out.insert(v.id);
                        collect_binders(r, out);
                    }
                }
            }
            collect_binders(body, out);
        }
        Expr::Case(scrut, alts, _, _) => {
            collect_binders(scrut, out);
            for alt in alts {
                for b in &alt.binders {
                    out.insert(b.id);
                }
                collect_binders(&alt.rhs, out);
            }
        }
        Expr::App(f, x, _) => {
            collect_binders(f, out);
            collect_binders(x, out);
        }
        Expr::TyApp(b, _, _)
        | Expr::TyLam(_, b, _)
        | Expr::Lazy(b, _)
        | Expr::Cast(b, _, _)
        | Expr::Tick(_, b, _) => collect_binders(b, out),
        _ => {}
    }
}

fn collect_body_tyvars(e: &Expr, resolved: &FxHashMap<Span, Ty>, out: &mut Vec<TyVar>) {
    match e {
        Expr::Var(v, span) => {
            collect_monad_tyvars(resolved.get(span).unwrap_or(&v.ty), out);
        }
        Expr::Lit(_, ty, _) | Expr::Type(ty, _) => collect_monad_tyvars(ty, out),
        Expr::App(f, x, _) => {
            collect_body_tyvars(f, resolved, out);
            collect_body_tyvars(x, resolved, out);
        }
        Expr::TyApp(f, ty, _) => {
            collect_body_tyvars(f, resolved, out);
            collect_monad_tyvars(ty, out);
        }
        Expr::Lam(v, b, _) => {
            collect_monad_tyvars(&v.ty, out);
            collect_body_tyvars(b, resolved, out);
        }
        Expr::TyLam(_, b, _) | Expr::Lazy(b, _) | Expr::Cast(b, _, _) | Expr::Tick(_, b, _) => {
            collect_body_tyvars(b, resolved, out);
        }
        Expr::Let(bind, body, _) => {
            match bind.as_ref() {
                Bind::NonRec(v, r) => {
                    collect_monad_tyvars(&v.ty, out);
                    collect_body_tyvars(r, resolved, out);
                }
                Bind::Rec(bs) => {
                    for (v, r) in bs {
                        collect_monad_tyvars(&v.ty, out);
                        collect_body_tyvars(r, resolved, out);
                    }
                }
            }
            collect_body_tyvars(body, resolved, out);
        }
        Expr::Case(s, alts, ty, _) => {
            collect_body_tyvars(s, resolved, out);
            collect_monad_tyvars(ty, out);
            for a in alts {
                for b in &a.binders {
                    collect_monad_tyvars(&b.ty, out);
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
                let base_ty = ctx.occ_ty(span).unwrap_or(&v.ty);
                unfold_newtypes(&subst.apply(base_ty), ctx.newtypes, 0)
            };
            // A usable occurrence type that is a ground transformer stack: specialize
            // the callee at the monad that type names.
            if !has_free_tyvar(&new_ty) && mentions_transformer(&new_ty) && !over_applied(&new_ty) {
                if let Some((ov, ob)) = ctx
                    .source_binding(v)
                    .filter(|_| !ctx.local_binders.contains(&v.id))
                {
                    if has_free_tyvar(&ov.ty) {
                        if let Some(inst) = instantiation_of(&ov.ty, &new_ty) {
                            if let Some(spec) = get_or_specialize(&ov, &ob, &inst, ctx) {
                                return Expr::Var(spec, *span);
                            }
                        }
                    }
                }
            } else if matches!(ctx.occ_ty(span).unwrap_or(&v.ty), Ty::Error)
                && mentions_transformer(&ctx.cur_monad)
            {
                // The occurrence type was ERASED (pandoc's `lift $ setupTranslations
                // meta` records `Ty::Error`), so there is no transformer to read from
                // it. But every polymorphic-monad call inside a clone specialized at
                // `cur_monad` runs in that same monad (the writer's inner monad IS it,
                // even under `lift`), so specialize the callee at `cur_monad`.
                let cur = ctx.cur_monad.clone();
                if let Some((ov, ob)) = ctx
                    .source_binding(v)
                    .filter(|_| !ctx.local_binders.contains(&v.id))
                {
                    if has_free_tyvar(&ov.ty) {
                        let inst = Instantiation::at_monad(cur);
                        if let Some(spec) = get_or_specialize(&ov, &ob, &inst, ctx) {
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
                if let Some(class) =
                    dict_class_name(d.name.as_str()).filter(|_| ctx.monad_dicts.contains(&d.id))
                {
                    if let Some(op) = class_sel_builtin(class, sel.name.as_str()) {
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
        Expr::Let(bind, body, span) => {
            let new_body = specialize_body(body, subst, ctx);
            // A dictionary `let` the selection rewrite above emptied out
            // (`let $superApplicative = $sel_0 $dMonad in pure`) is dropped, so
            // the builtin operator is applied DIRECTLY: codegen routes `pure 1`
            // by the clone's transformer stack, but a `pure` reached as a let
            // body is lowered as a value — IO's `pure`, whose result (the bare
            // `1`) the StateT caller then ran as a closure.
            if let Bind::NonRec(x, _) = bind.as_ref() {
                if dict_class_name(x.name.as_str()).is_some()
                    && ctx.monad_dicts.contains(&x.id)
                    && !new_body.free_vars().iter().any(|v| v.id == x.id)
                {
                    return new_body;
                }
            }
            Expr::Let(
                Box::new(specialize_bind(bind, subst, ctx)),
                Box::new(new_body),
                *span,
            )
        }
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
pub(crate) fn match_ty(pat: &Ty, conc: &Ty, subst: &mut Subst) -> bool {
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
/// The class a dictionary variable's name denotes — a direct `C =>` parameter
/// (`$dC_NNN`) or a superclass dictionary extracted from another class's
/// dictionary (`$superC_NNN`, named by `select_method_via_superclass`). Matched
/// EXACTLY: `$dPandocMonad` is `PandocMonad` (whose `$sel_1` is the `Applicative`
/// superclass, NOT `>>=`), not `Monad`; the constructed instance dicts named
/// `$dict...` are not `$d`/`$super` params and never match.
fn dict_class_name(name: &str) -> Option<&str> {
    let rest = name
        .strip_prefix("$super")
        .or_else(|| name.strip_prefix("$d"))?;
    // The class name runs up to the fresh-id suffix `_NNN` (or the whole rest
    // when there is none). Reject the empty match so `$d`/`$super` alone fail.
    let cls = rest.split('_').next().filter(|s| !s.is_empty())?;
    Some(cls)
}

/// Map a `(class, field selector)` on a null-placeholder superclass/param dict to
/// the plain builtin operator, which codegen then routes by the concrete
/// transformer stack. The field index is `superclasses.len() + method_index` of
/// the class (see `select_method_via_superclass`): `Monad [Applicative,(>>=),(>>)]`,
/// `Applicative [Functor,pure,(<*>)]`, `Functor [fmap,(<$)]`. Only the do-notation
/// / `Applicative`/`Functor` operators a specialized clone actually needs are
/// mapped; anything else is left as a genuine dictionary selection.
fn class_sel_builtin(class: &str, sel: &str) -> Option<&'static str> {
    match (class, sel) {
        ("Monad", "$sel_1") => Some(">>="),
        ("Monad", "$sel_2") => Some(">>"),
        ("Applicative", "$sel_1") => Some("pure"),
        ("Applicative", "$sel_2") => Some("<*>"),
        ("Functor", "$sel_0") => Some("fmap"),
        _ => None,
    }
}

/// The ultimate result of a function type (all leading `->` arrows stripped).
pub(crate) fn ultimate_result(ty: &Ty) -> &Ty {
    let mut r = ty;
    while let Ty::Fun(_, b) = r {
        r = b;
    }
    r
}

pub(crate) fn has_free_tyvar(ty: &Ty) -> bool {
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
        Ty::Con(c) => is_transformer_name(c.name.as_str()),
        Ty::App(a, b) | Ty::Fun(a, b) => mentions_transformer(a) || mentions_transformer(b),
        Ty::List(a) => mentions_transformer(a),
        Ty::Tuple(ts) => ts.iter().any(mentions_transformer),
        Ty::Forall(_, b) => mentions_transformer(b),
        _ => false,
    }
}

/// The type variables of `ty` in MONAD position: the head of an application
/// (`m` in `m a`) or the inner-monad argument of a transformer (`m` in
/// `StateT s m a`).
fn collect_monad_tyvars(ty: &Ty, out: &mut Vec<TyVar>) {
    let push = |v: &TyVar, out: &mut Vec<TyVar>| {
        if !out.contains(v) {
            out.push(v.clone());
        }
    };
    match ty {
        Ty::App(..) => {
            let mut head = ty;
            let mut args = Vec::new();
            while let Ty::App(f, x) = head {
                args.push(x.as_ref());
                head = f.as_ref();
            }
            args.reverse();
            match head {
                Ty::Var(v) => push(v, out),
                Ty::Con(c) if is_transformer_name(c.name.as_str()) => {
                    if let Some(Ty::Var(v)) = args.get(1) {
                        push(v, out);
                    }
                }
                _ => {}
            }
            for a in args {
                collect_monad_tyvars(a, out);
            }
        }
        Ty::Fun(a, b) => {
            collect_monad_tyvars(a, out);
            collect_monad_tyvars(b, out);
        }
        Ty::List(a) => collect_monad_tyvars(a, out),
        Ty::Tuple(ts) => ts.iter().for_each(|t| collect_monad_tyvars(t, out)),
        Ty::Forall(_, b) => collect_monad_tyvars(b, out),
        _ => {}
    }
}

pub(crate) fn collect_tyvars(ty: &Ty, out: &mut Vec<TyVar>) {
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
    fn class_selectors_map_to_operators() {
        assert_eq!(class_sel_builtin("Monad", "$sel_1"), Some(">>="));
        assert_eq!(class_sel_builtin("Monad", "$sel_2"), Some(">>"));
        assert_eq!(class_sel_builtin("Applicative", "$sel_1"), Some("pure"));
        assert_eq!(class_sel_builtin("Applicative", "$sel_2"), Some("<*>"));
        assert_eq!(class_sel_builtin("Functor", "$sel_0"), Some("fmap"));
        // Not a mapped op / wrong class.
        assert_eq!(class_sel_builtin("Monad", "$sel_0"), None);
        assert_eq!(class_sel_builtin("PandocMonad", "$sel_1"), None);
        assert_eq!(class_sel_builtin("MonadError", "$sel_1"), None);
    }

    fn v(name: &str, id: usize) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id), Ty::Error)
    }
    fn sel(n: usize, d: Expr) -> Expr {
        Expr::App(
            Box::new(Expr::Var(v(&format!("$sel_{n}"), 0), Span::DUMMY)),
            Box::new(d),
            Span::DUMMY,
        )
    }

    #[test]
    fn own_monad_dicts_takes_params_and_superclass_selections() {
        // \$dPandocMonad -> let $superMonad = $sel_0 $dPandocMonad in
        //                     let $dMonad_c = <constructed> in $sel_1 $superMonad
        let param = v("$dPandocMonad_1", 1);
        let supr = v("$superMonad_2", 2);
        let constructed = v("$dMonad_3", 3);
        let body = Expr::Lam(
            param.clone(),
            Box::new(Expr::Let(
                Box::new(Bind::NonRec(
                    supr.clone(),
                    Box::new(sel(0, Expr::Var(param.clone(), Span::DUMMY))),
                )),
                Box::new(Expr::Let(
                    Box::new(Bind::NonRec(
                        constructed.clone(),
                        // A dictionary the body BUILDS: not derived from the param.
                        Box::new(Expr::Lit(crate::Literal::Int(0), Ty::Error, Span::DUMMY)),
                    )),
                    Box::new(sel(1, Expr::Var(supr.clone(), Span::DUMMY))),
                    Span::DUMMY,
                )),
                Span::DUMMY,
            )),
            Span::DUMMY,
        );
        let dicts = own_monad_dicts(&body);
        assert!(dicts.contains(&param.id), "the dictionary parameter");
        assert!(dicts.contains(&supr.id), "a superclass selection from it");
        assert!(
            !dicts.contains(&constructed.id),
            "a dictionary the body constructed for another monad must NOT count — \
             rewriting its selections to builtins ran pandoc's parser through the \
             writer's transformer stack"
        );
    }

    #[test]
    fn dict_class_name_extracts_class_exactly() {
        // Direct params and extracted superclass dicts.
        assert_eq!(dict_class_name("$dMonad_1234"), Some("Monad"));
        assert_eq!(dict_class_name("$dMonad"), Some("Monad"));
        assert_eq!(dict_class_name("$superMonad_5678"), Some("Monad"));
        assert_eq!(dict_class_name("$superApplicative_2"), Some("Applicative"));
        assert_eq!(dict_class_name("$dPandocMonad_9"), Some("PandocMonad"));
        assert_eq!(dict_class_name("$dMonadError_1"), Some("MonadError"));
        // Constructed instance dicts are not `$d`/`$super` params.
        assert_eq!(dict_class_name("$dictMonad_PandocIO_1"), Some("ictMonad"));
        // Non-dict names.
        assert_eq!(dict_class_name("modify"), None);
        assert_eq!(dict_class_name("$super"), None);
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
