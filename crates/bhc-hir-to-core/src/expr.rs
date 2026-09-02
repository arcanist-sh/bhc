//! Expression lowering from HIR to Core.
//!
//! This module handles the transformation of HIR expressions to Core
//! expressions. Key transformations include:
//!
//! - `If` expressions become `Case` on booleans
//! - `Lam` with multiple patterns becomes nested lambdas with case
//! - `Tuple` and `List` become constructor applications

use bhc_core::{self as core, Alt, AltCon, Bind, DataCon, Literal, Var, VarId};
use bhc_hir::{self as hir, DefId, DefRef, Expr, Lit};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Kind, Ty, TyCon};

use crate::context::{has_type_variables, LowerContext};
use crate::pattern::lower_pat_to_alt;
use crate::LowerResult;

/// Format a constraint for error messages.
fn format_constraint(constraint: &Constraint) -> String {
    if constraint.args.is_empty() {
        constraint.class.as_str().to_string()
    } else {
        format!(
            "{} {}",
            constraint.class.as_str(),
            constraint
                .args
                .iter()
                .map(format_type)
                .collect::<Vec<_>>()
                .join(" ")
        )
    }
}

/// Format a type for error messages.
fn format_type(ty: &Ty) -> String {
    match ty {
        Ty::Con(c) => c.name.as_str().to_string(),
        Ty::Var(v) => format!("t{}", v.id),
        Ty::App(f, a) => format!("({} {})", format_type(f), format_type(a)),
        Ty::Fun(a, r) => format!("({} -> {})", format_type(a), format_type(r)),
        Ty::Tuple(ts) => format!(
            "({})",
            ts.iter().map(format_type).collect::<Vec<_>>().join(", ")
        ),
        Ty::List(e) => format!("[{}]", format_type(e)),
        _ => "?".to_string(),
    }
}

/// Create an error expression that will fail at runtime with a message.
fn make_error_expr(msg: &str, span: Span) -> core::Expr {
    let error_var = Var {
        name: Symbol::intern("error"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let msg_lit = core::Expr::Lit(Literal::String(Symbol::intern(msg)), Ty::Error, span);
    core::Expr::App(
        Box::new(core::Expr::Var(error_var, span)),
        Box::new(msg_lit),
        span,
    )
}

/// Lower a HIR expression to Core.
pub fn lower_expr(ctx: &mut LowerContext, expr: &hir::Expr) -> LowerResult<core::Expr> {
    let span = expr.span();
    let core = lower_expr_inner(ctx, expr)?;
    Ok(annotate_ty(ctx, span, core))
}

/// Populate a lowered Core `Var`'s type slot from the type checker's per-node
/// types (spec/BHC-BRIEF-0002, Task 2), keyed by source span. `Expr::ty()` is
/// already compositional, so filling a function-typed `Var` (e.g. the `f` in
/// `map f xs`) makes `f.ty()` return a real `Fun(elem, acc)` instead of
/// `Fun(Error, Error)` — which is exactly what the numeric fusion rewrites gate
/// on (`fuse::try_fuse_sum_map` reads the mapped function's codomain).
///
/// Scope is deliberately narrow. We annotate **only `Var` nodes whose recorded
/// type is a function type**, for two reasons:
///
///   1. That is the only shape the simplifier's fusion gate inspects.
///   2. Codegen was written assuming Core carries `Ty::Error` and reads leaf
///      *scalar* types to pick integer widths. Populating scalar `Lit`/`Var`
///      types feeds it data it never consumed, and mismatched widths surface as
///      LLVM `icmp` type errors (e.g. `icmp eq i32 %c, i64 58` for a char-code
///      compare). Function types carry no width, so annotating them is inert for
///      codegen. Full leaf population is future work gated on codegen learning
///      to consume these types.
///
/// Conservative: only fills a slot that is currently `Ty::Error`, so anything
/// lowering already typed correctly (e.g. constructor schemes) is preserved.
/// Whether a variable of this type may carry it into Core.
///
/// Function types have always been annotated. ADTs are added because `show` on
/// a variable had no way to find its type at all: `build_show_descriptor` falls
/// back to `expr.ty()`, a Core variable carries `Ty::Error`, and the default
/// coercion is Int — so `let n = R 97 in show n` printed the POINTER as a
/// number while `show (R 97)` printed `R 97`. Any variable was affected, not
/// just a monadically bound one.
///
/// Scalars stay out. Codegen infers value widths assuming `Ty::Error`, and
/// handing it a `Char` makes it emit `icmp eq i32 %c, i64 32`, which fails LLVM
/// verification — the `milestone_e_json` fixture catches exactly that. An ADT
/// is a pointer either way, so it carries no width to disagree about.
fn ty_is_annotatable(ty: &Ty) -> bool {
    fn head(t: &Ty) -> &Ty {
        match t {
            Ty::App(f, _) => head(f),
            other => other,
        }
    }
    if matches!(ty, Ty::Fun(_, _)) {
        return true;
    }
    // Lists and tuples are pointers too, and a bound list had the same problem
    // an ADT did: `let xs = ["ab"] in print xs` printed the LIST's address.
    //
    // Only when the ELEMENT type is known, though. A `String` is `[Char]`, and
    // annotating one puts a Char back in front of codegen's width inference —
    // `icmp eq i32 %to_char, i64 44`, which fails LLVM verification; the
    // `milestone_e_json` fixture catches that. Worse, typeck records the
    // element of a String it has not resolved as a plain type VARIABLE, and
    // `[a]` names nothing codegen can act on: the WASM backend read the C
    // string behind `getLine` as a cons list, so
    // `putStrLn ("Sum for " ++ label ++ ":")` printed `Sum for p` and a NUL
    // byte. A list OF strings is still fine — only the outer type is
    // annotated, and its elements are described by the show descriptor.
    if let Some(elem) = list_element(ty) {
        return match elem {
            Ty::Con(tc) => tc.name.as_str() != "Char",
            Ty::List(_) | Ty::Tuple(_) | Ty::App(_, _) => true,
            _ => false,
        };
    }
    if matches!(ty, Ty::Tuple(_)) {
        return true;
    }
    match head(ty) {
        Ty::Con(tc) => !matches!(
            tc.name.as_str(),
            "Int"
                | "Integer"
                | "Char"
                | "Double"
                | "Float"
                | "Word"
                | "Bool"
                | "String"
                | "Ordering"
                | "IO"
        ),
        _ => false,
    }
}

/// The element of a list type, whichever way it is spelled: `Ty::List` is sugar
/// for `App(Con "[]", elem)` and both reach here.
fn list_element(ty: &Ty) -> Option<&Ty> {
    match ty {
        Ty::List(elem) => Some(elem.as_ref()),
        Ty::App(f, elem) => match f.as_ref() {
            Ty::Con(tc) if tc.name.as_str() == "[]" => Some(elem.as_ref()),
            _ => None,
        },
        _ => None,
    }
}

/// The monad-family methods whose Core type is read ONLY to choose a
/// transformer implementation, never to size a value.
fn is_monad_method_name(name: &str) -> bool {
    matches!(
        name,
        ">>=" | ">>" | "=<<" | "return" | "pure" | "fmap" | "<*>" | "<$>"
    )
}

fn annotate_ty(ctx: &LowerContext, span: bhc_span::Span, core: core::Expr) -> core::Expr {
    // A monad method takes the RESOLVED occurrence type. Its raw one is
    // whatever a single pass left behind — `foldl' (>>=) …` records
    // `(t606 t604) -> ((t604 -> t606 t608) -> t606 t608)`, a bare type
    // VARIABLE for the monad — while the resolved map has
    // `ExceptT e IO Int -> …`. Codegen reads exactly this to pick the
    // transformer implementation, and with a variable it falls back to IO:
    // `foldl' (>>=) (return 0) [step]` at ExceptT ran the action, got an
    // `Either`, and passed THAT to the continuation as if it were the value.
    //
    // Deliberately confined to these names. The resolved map is documented as
    // dispatch-only — typing every Core node from it regresses codegen widths —
    // and a monad method's type is never used for a width.
    if let core::Expr::Var(ref v, _) = core {
        if matches!(v.ty, Ty::Error) && is_monad_method_name(v.name.as_str()) {
            if let Some(resolved) = ctx.resolved_expr_ty_opt(span) {
                if ty_is_annotatable(&resolved) {
                    if let core::Expr::Var(mut v, s) = core {
                        v.ty = resolved;
                        return core::Expr::Var(v, s);
                    }
                    unreachable!("matched Var above")
                }
            }
        }
    }
    let Some(ty) = ctx.expr_ty_opt(span) else {
        return core;
    };
    if !ty_is_annotatable(&ty) {
        return core;
    }
    match core {
        core::Expr::Var(mut v, s) => {
            if matches!(v.ty, Ty::Error) {
                v.ty = ty;
            }
            core::Expr::Var(v, s)
        }
        other => other,
    }
}

fn lower_expr_inner(ctx: &mut LowerContext, expr: &hir::Expr) -> LowerResult<core::Expr> {
    match expr {
        Expr::Lit(lit, span) => lower_lit_at(ctx, lit, *span),

        Expr::Var(def_ref) => lower_var(ctx, def_ref),

        Expr::Con(def_ref) => lower_con(ctx, def_ref),

        Expr::App(f, x, span) => lower_app(ctx, f, x, *span),

        Expr::Lam(pats, body, span) => lower_lambda(ctx, pats, body, *span),

        Expr::Let(bindings, body, span) => lower_let(ctx, bindings, body, *span),

        Expr::Case(scrutinee, alts, span) => lower_case(ctx, scrutinee, alts, *span),

        Expr::If(cond, then_br, else_br, span) => lower_if(ctx, cond, then_br, else_br, *span),

        Expr::Tuple(elems, span) => lower_tuple(ctx, elems, *span),

        Expr::List(elems, span) => lower_list(ctx, elems, *span),

        Expr::Record(con_ref, fields, span) => lower_record(ctx, con_ref, fields, *span),

        Expr::FieldAccess(expr, field, span) => lower_field_access(ctx, expr, *field, *span),

        Expr::RecordUpdate(expr, fields, span) => lower_record_update(ctx, expr, fields, *span),

        Expr::Ann(expr, ty, span) => {
            // Check if this is an Integer-annotated literal — if so, create Literal::Integer
            if let Expr::Lit(Lit::Int(n), _) = expr.as_ref() {
                if matches!(ty, Ty::Con(tc) if tc.name.as_str() == "Integer") {
                    return Ok(core::Expr::Lit(
                        Literal::Integer(*n),
                        Ty::Con(TyCon::new(Symbol::intern("Integer"), Kind::Star)),
                        *span,
                    ));
                }
            }
            // Check for negated Integer literal: negate (Int n) :: Integer
            if let Expr::App(f, arg, _) = expr.as_ref() {
                if let Expr::Var(def_ref) = f.as_ref() {
                    let is_negate = ctx
                        .lookup_var(def_ref.def_id)
                        .is_some_and(|v| v.name.as_str() == "negate");
                    if is_negate {
                        if let Expr::Lit(Lit::Int(n), _) = arg.as_ref() {
                            if matches!(ty, Ty::Con(tc) if tc.name.as_str() == "Integer") {
                                return Ok(core::Expr::Lit(
                                    Literal::Integer(-*n),
                                    Ty::Con(TyCon::new(Symbol::intern("Integer"), Kind::Star)),
                                    *span,
                                ));
                            }
                        }
                    }
                }
            }
            // Type annotations are erased in Core (types are tracked separately)
            lower_expr(ctx, expr)
        }

        Expr::TypeApp(expr, ty, span) => lower_type_app(ctx, expr, ty, *span),

        Expr::Error(span) => {
            // Generate a runtime error expression
            let error_name = Symbol::intern("error");
            let error_var = Var {
                name: error_name,
                id: VarId::new(0),
                ty: Ty::Error,
            };
            let msg = core::Expr::Lit(
                Literal::String(Symbol::intern("pattern match error")),
                Ty::Error,
                *span,
            );
            Ok(core::Expr::App(
                Box::new(core::Expr::Var(error_var, *span)),
                Box::new(msg),
                *span,
            ))
        }
    }
}

/// Lower a literal, packing a STRING literal that is used at type `Text`.
///
/// bhc's `Text` is a real struct — `bhc_text_pack` walks a char list and
/// allocates a `{data ptr, offset, byte_len}` header, and every `Data.Text.*`
/// RTS entry point reads that header. A string literal is a char list, so an
/// OverloadedStrings literal at type `Text` was handed to those functions
/// unconverted: `TIO.putStrLn ("hello" :: Text)` printed an empty line and
/// `T.length` on it segfaulted. `fromString` is gone by Core (it lowers to
/// identity, which is right for `String`), so the conversion has to key off the
/// literal's own occurrence type.
fn lower_lit_at(ctx: &LowerContext, lit: &Lit, span: Span) -> LowerResult<core::Expr> {
    let core_lit = lower_lit(lit, span)?;
    if !matches!(lit, Lit::String(_)) {
        return Ok(core_lit);
    }
    match refined_occurrence_ty(ctx, span) {
        Some(ty) if ty_head_is(&ty, "Text") => {
            let pack = Var {
                name: Symbol::intern("Data.Text.pack"),
                id: VarId::new(0),
                ty: Ty::Error,
            };
            Ok(core::Expr::App(
                Box::new(core::Expr::Var(pack, span)),
                Box::new(core_lit),
                span,
            ))
        }
        _ => Ok(core_lit),
    }
}

/// Whether a type's head constructor is named `name`.
fn ty_head_is(ty: &Ty, name: &str) -> bool {
    match ty {
        Ty::Con(c) => c.name.as_str() == name,
        Ty::App(f, _) => ty_head_is(f, name),
        _ => false,
    }
}

/// Lower a literal to Core.
fn lower_lit(lit: &Lit, span: Span) -> LowerResult<core::Expr> {
    let core_lit = match lit {
        Lit::Int(n) => Literal::Int(*n as i64),
        Lit::Float(f) => Literal::Float(*f),
        Lit::Char(c) => Literal::Char(*c),
        Lit::String(s) => Literal::String(*s),
    };
    Ok(core::Expr::Lit(core_lit, Ty::Error, span))
}

/// From a monad-family method occurrence's instantiated type, recover the
/// monad/functor constructor `m` it is used at. For `<|> :: P a -> P a -> P a`
/// the first parameter is `P a` — strip one `App` to get `P`'s head
/// application; for a nullary/result-only method (`mzero :: P a`) use the
/// result type itself. Returns None when the shape doesn't match (e.g. the
/// occurrence type is still a bare variable), letting the caller fall through.
/// Whether a type's head is a concrete constructor — dispatching an instance
/// at a bare type VARIABLE would unify with any instance (the first match),
/// turning a constraint-dict method like `Future`'s `mempty = return mempty`
/// into a self-referential loop.
fn has_concrete_head(ty: &Ty) -> bool {
    match ty {
        Ty::Con(_) | Ty::List(_) | Ty::Tuple(_) | Ty::Prim(_) => true,
        Ty::App(f, _) => has_concrete_head(f),
        _ => false,
    }
}

/// Whether the binding being lowered RETURNS in the monad an in-scope
/// dictionary for `class_name` is for.
///
/// This is the discriminator for a value-position monad method, where the
/// occurrence itself carries no usable type. `runParsecT :: Monad m => … -> m
/// (Consumed …)` returns in `m`, the dictionary's own monad, so a bare
/// `return` there means that monad's `pure`. `poly :: Monad m => ParsecT
/// String () m SourcePos` returns in ParsecT, so its `return` is ParsecT's —
/// routing that one to the INNER monad's dictionary is the same wrong-monad
/// selection `in_scope_dict_matches` exists to prevent, and it segfaults the
/// parser probes.
fn binding_returns_in_dict_monad(ctx: &LowerContext, class_name: Symbol) -> bool {
    fn result_of(t: &Ty) -> &Ty {
        match t {
            Ty::Fun(_, r) => result_of(r),
            other => other,
        }
    }
    let Some(dict_ty) = ctx.lookup_dict_ty(class_name).cloned() else {
        return false;
    };
    ctx.current_binding_sig()
        .is_some_and(|sig| same_type_head(&dict_ty, result_of(sig)))
}

/// Whether an in-scope dictionary for `class_name` is one this occurrence
/// should actually be selecting a method from.
///
/// `lookup_dict` matches by CLASS ALONE, so it returns a dictionary for any
/// type. That is fine for a user class, where at most one is in scope. It is
/// wrong for the monad family once a binding constrained over its OWN monad
/// puts a `Monad m` dictionary in scope: parsec's `getPosition :: Monad m =>
/// ParsecT s u m SourcePos` selected its do-block's `>>=` out of the dictionary
/// for ParsecT's INNER monad, so every parser ran from a garbage state and
/// `getPosition` reported column 0.
///
/// The default when the occurrence's monad cannot be determined is to REFUSE.
/// Before monad-family constraints became dictionary-passed, no such dictionary
/// was ever in scope and these methods went to the concrete-instance dispatch
/// below; refusing restores exactly that behaviour instead of guessing.
fn in_scope_dict_matches(ctx: &LowerContext, class_name: Symbol, span: Span) -> bool {
    let Some(dict_ty) = ctx.lookup_dict_ty(class_name).cloned() else {
        // No recorded type: this dictionary predates the witness (an
        // existential pattern match, say), so keep the old behaviour.
        return true;
    };
    let occ_head = refined_occurrence_ty(ctx, span)
        .as_ref()
        .and_then(monad_head_of_method_occurrence);
    match occ_head {
        // A dictionary for one monad must not answer for another, whatever
        // the class. `readWithM p` runs `p` at `ParsecT Sources st m` inside a
        // `PandocMonad m =>` binding, so pandoc's `getCommonState` — a
        // PandocMonad method the ParsecT instance defines as `lift
        // getCommonState` — was selected from the dictionary for `m` and the
        // parser was handed an ExceptT-over-StateT action to run.
        Some(occ_head) => same_type_head(&dict_ty, &occ_head),
        // Nothing recorded: keep what each class kind did before.
        None => !ctx.is_monad_family_class(class_name),
    }
}

/// Whether two types name the same monad, comparing only head constructors —
/// `ParsecT s u m` and `ParsecT [Char] () IO` are the same monad, and a type
/// variable matches only another variable.
pub(crate) fn same_type_head(a: &Ty, b: &Ty) -> bool {
    fn head(ty: &Ty) -> &Ty {
        match ty {
            Ty::App(f, _) => head(f),
            other => other,
        }
    }
    match (head(a), head(b)) {
        (Ty::Con(x), Ty::Con(y)) => x.name == y.name,
        (Ty::Var(_), Ty::Var(_)) => true,
        _ => false,
    }
}

pub(crate) fn monad_head_of_method_occurrence(ty: &Ty) -> Option<Ty> {
    fn applied_to(ty: &Ty) -> Option<Ty> {
        match ty {
            Ty::App(m, _) => Some(m.as_ref().clone()),
            _ => None,
        }
    }
    fn result_of(ty: &Ty) -> &Ty {
        match ty {
            Ty::Fun(_, r) => result_of(r),
            t => t,
        }
    }
    let from_first_arg = match ty {
        Ty::Fun(a, _) => applied_to(a),
        t => applied_to(t),
    };
    // `>>=` and friends carry the monad in their first argument; a method
    // like `getsCommonState :: (CommonState -> a) -> m a` carries it only in
    // the RESULT, and without that the occurrence said nothing and the
    // dictionary in scope answered for a monad that was not the use site's.
    from_first_arg.or_else(|| applied_to(result_of(ty)))
}

/// Lower a variable reference to Core.
///
/// This handles several cases:
///
/// 1. **Class methods**: If the variable is a class method (like `==` from `Eq`),
///    and we're inside a constrained function, select the method from the
///    appropriate dictionary.
///
/// 2. **Constrained functions**: If the referenced function has type class
///    constraints, apply dictionary arguments from the current scope or
///    resolve instances for concrete types.
///
/// 3. **Regular variables**: Just return a variable reference.
fn lower_var(ctx: &mut LowerContext, def_ref: &DefRef) -> LowerResult<core::Expr> {
    // First, check if this is a class method reference
    let var_name = ctx.lookup_var(def_ref.def_id).map(|v| v.name);

    if std::env::var_os("BHC_DBG_MONOP").is_some() {
        if let Some(n) = var_name {
            if matches!(n.as_str(), ">>=" | ">>" | "=<<" | "return" | "pure") {
                eprintln!(
                    "MONOP {} raw={:?} resolved={:?} refined={:?}",
                    n.as_str(),
                    ctx.expr_ty_opt(def_ref.span).map(|t| dbg_ty(&t)),
                    ctx.resolved_expr_ty_opt(def_ref.span).map(|t| dbg_ty(&t)),
                    refined_occurrence_ty(ctx, def_ref.span).map(|t| dbg_ty(&t)),
                );
            }
        }
    }

    if let Some(name) = var_name {
        // `guard :: Alternative f => Bool -> f ()` — an external constrained
        // FUNCTION (not a class method), unimplemented as a builtin, so it
        // stubs at runtime (pandoc's `guardEnabled` = `getOption
        // readerExtensions >>= guard . extensionEnabled ext`). When the
        // occurrence type pins `f` to a user monad, synthesize its body
        // `\b -> if b then pure () else empty` with both methods resolved at
        // that instance.
        // `ap :: Monad m => m (a -> b) -> m a -> m b` in VALUE position —
        // parsec writes `instance Applicative (ParsecT s u m) where (<*>) = ap`.
        // codegen has no implementation and emits `stub: ap not implemented`,
        // which is where `readMarkdown` stopped once its parsers ran at all.
        if name.as_str() == "ap" {
            // The occurrence type when typeck recorded one, else the instance
            // being defined — `(<*>) = ap` is a bare Var with no type of its
            // own, and it is exactly the case that matters.
            let m_ty = match ctx.resolved_expr_ty_opt(def_ref.span) {
                Some(Ty::Fun(_, r1)) => match r1.as_ref() {
                    Ty::Fun(_, r2) => match r2.as_ref() {
                        Ty::App(m, _) => Some(m.as_ref().clone()),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            }
            .or_else(|| ctx.current_instance_type().cloned());
            if let Some(m_ty) = m_ty {
                if !monad_runs_eagerly(&m_ty) && applied_head_name(&m_ty).is_some() {
                    if let Some(lam) = lower_ap_lambda(ctx, &m_ty, def_ref.span) {
                        return Ok(lam);
                    }
                }
            }
        }
        if name.as_str() == "guard" {
            if let Some(Ty::Fun(_, r)) = ctx.resolved_expr_ty_opt(def_ref.span) {
                if let Ty::App(f_ty, _) = r.as_ref() {
                    if has_concrete_head(f_ty) && !LowerContext::is_builtin_monad_type(f_ty) {
                        let pure_e = ctx.resolve_method_at_concrete_type(
                            Symbol::intern("pure"),
                            Symbol::intern("Applicative"),
                            f_ty,
                            def_ref.span,
                        );
                        let empty_e = ctx.resolve_method_at_concrete_type(
                            Symbol::intern("empty"),
                            Symbol::intern("Alternative"),
                            f_ty,
                            def_ref.span,
                        );
                        if let (Some(pure_e), Some(empty_e)) = (pure_e, empty_e) {
                            let b = ctx.fresh_var("$guard_b", Ty::Error, def_ref.span);
                            let unit = core::Expr::Var(
                                Var {
                                    name: Symbol::intern("()"),
                                    id: VarId::new(0),
                                    ty: Ty::Error,
                                },
                                def_ref.span,
                            );
                            let body = make_if_expr(
                                core::Expr::Var(b.clone(), def_ref.span),
                                core::Expr::App(Box::new(pure_e), Box::new(unit), def_ref.span),
                                empty_e,
                                def_ref.span,
                            );
                            return Ok(core::Expr::Lam(b, Box::new(body), def_ref.span));
                        }
                    }
                }
            }
        }
        // `mappend`/`<>` at Ordering (parsec's `compareErrorPos x y =
        // Mon.mappend (compare …) (compare …)`): Ordering has no Semigroup
        // instance registered anywhere, so the occurrence stubbed at runtime.
        // Synthesize the Ordering semigroup: `\x y -> case x of EQ -> y;
        // _ -> x`.
        if matches!(name.as_str(), "mappend" | "<>") {
            let is_ordering = |t: &Ty| matches!(t, Ty::Con(tc) if tc.name.as_str() == "Ordering");
            if let Some(Ty::Fun(a, r)) = ctx.resolved_expr_ty_opt(def_ref.span) {
                if let Ty::Fun(b, res) = r.as_ref() {
                    if is_ordering(&a) && is_ordering(b) && is_ordering(res) {
                        let span = def_ref.span;
                        let x = ctx.fresh_var("$ord_x", Ty::Error, span);
                        let y = ctx.fresh_var("$ord_y", Ty::Error, span);
                        let eq_con = DataCon {
                            name: Symbol::intern("EQ"),
                            ty_con: TyCon::new(Symbol::intern("Ordering"), Kind::Star),
                            tag: 1,
                            arity: 0,
                        };
                        let case = core::Expr::Case(
                            Box::new(core::Expr::Var(x.clone(), span)),
                            vec![
                                Alt {
                                    con: AltCon::DataCon(eq_con),
                                    binders: vec![],
                                    rhs: core::Expr::Var(y.clone(), span),
                                },
                                Alt {
                                    con: AltCon::Default,
                                    binders: vec![],
                                    rhs: core::Expr::Var(x.clone(), span),
                                },
                            ],
                            Ty::Error,
                            span,
                        );
                        let inner = core::Expr::Lam(y, Box::new(case), span);
                        return Ok(core::Expr::Lam(x, Box::new(inner), span));
                    }
                }
            }
        }
        // A QUALIFIED reference to a monad-family operator inside an instance
        // body means the ENCLOSING instance's method. parsec writes `(>>) =
        // (Applicative.*>)` in `instance Monad (ParsecT s u m)`, and
        // `Control.Applicative` is an unimplemented external, so `>>` at every
        // parser called a stub — pandoc's `readMarkdown` aborted in the first
        // `do` block it reached. Drop the qualifier and let the dispatch below
        // find the instance's own `*>`.
        let name = instance_local_qualified_method(ctx, name).unwrap_or(name);
        // Check if this is a class method
        let is_method = ctx.is_class_method(name);
        // `return` is in NO class's method list: the builtin Monad layout is
        // [`>>=`, `>>`] and `return` is Applicative's `pure` under another
        // name, so `is_class_method` does not recognise it and the whole
        // dictionary-selection path below is skipped. An APPLIED `return` is
        // fine — `lower_app` dispatches it — but a VALUE occurrence fell
        // through to a builtin and codegen guessed an ambient transformer
        // layer for it. parsec's `runParsecT` writes `return . Consumed .
        // return`, so its `return`s went bare and StateT's bind was handed a
        // non-closure to call.
        //
        // Route a value-position `return` through the superclass hop, which
        // already remaps it to Applicative's `pure`. Adding `return` to
        // Monad's method list instead would shift every Monad dictionary's
        // field indices, since methods are laid out after the superclass
        // slots.
        if is_method.is_none()
            && name.as_str() == "return"
            && crate::dictionary::monad_witness_enabled()
            && binding_returns_in_dict_monad(ctx, Symbol::intern("Monad"))
        {
            if let Some(method_expr) =
                ctx.select_method_via_superclass(Symbol::intern("Monad"), name, def_ref.span)
            {
                return Ok(method_expr);
            }
        }
        if let Some(class_name) = is_method {
            // This is a class method - we need to select it from a dictionary
            // Look for an in-scope dictionary for this class
            let dict_matches = in_scope_dict_matches(ctx, class_name, def_ref.span);
            if let Some(dict_var) = ctx.lookup_dict(class_name).filter(|_| dict_matches) {
                // Select the method from the dictionary
                if let Some(method_expr) =
                    ctx.select_method_from_dict(dict_var, class_name, name, def_ref.span)
                {
                    return Ok(method_expr);
                }
            } else if !ctx.is_user_class(class_name)
                && matches!(class_name.as_str(), "Semigroup" | "Monoid")
            {
                // The builtin VALUE classes. `mempty` has no argument to
                // dispatch on and no dictionary in scope at a top-level use,
                // and this arm previously required a USER class — so
                // `mempty :: [Int]` never reached instance resolution at all
                // and stayed a bare `mempty` Var in Core, which `length` then
                // walked as garbage. Only the result-type channel is opened
                // here; superclass extraction stays user-classes-only.
                if let Some(method_expr) =
                    ctx.select_method_by_result_type(class_name, name, def_ref.def_id, def_ref.span)
                {
                    return Ok(method_expr);
                }
            } else if class_name.as_str() == "MonadTrans" {
                // `lift` into a USER transformer. Without this it falls through
                // to the ambient-layer builtin, which for anything that is not
                // a known layer means `TransformerLayer::IO` — where lift is
                // identity, so the lifted action is handed on unwrapped and
                // parsec's bind then runs it as a parser.
                if let Some(method_expr) = ctx.select_monad_trans_method(name, def_ref.span) {
                    return Ok(method_expr);
                }
            } else if ctx.is_user_class(class_name) {
                // When the occurrence's own monad is KNOWN to differ from the
                // dictionary in scope, the instance is the answer and a
                // superclass hop is not — every hop reaches that same wrong
                // dictionary. `readWithM p` runs `p` at `ParsecT Sources st m`
                // inside a `PandocMonad m =>` binding, so pandoc's
                // `getCommonState` was selected from the dictionary for `m`
                // and the parser was handed an ExceptT-over-StateT action.
                if !dict_matches && ctx.lookup_dict(class_name).is_some() {
                    if let Some(method_expr) = ctx.select_method_by_result_type(
                        class_name,
                        name,
                        def_ref.def_id,
                        def_ref.span,
                    ) {
                        return Ok(method_expr);
                    }
                }
                // No direct dict in scope — try superclass extraction.
                // If we have MyOrd in scope and need MyEq, extract MyEq from MyOrd.
                if let Some(method_expr) =
                    ctx.select_method_via_superclass(class_name, name, def_ref.span)
                {
                    return Ok(method_expr);
                }
                // Result-position method (`def :: Default a => a`): no
                // argument will ever drive resolution, but typeck recorded
                // this use's resolved type by span — dispatch directly.
                if let Some(method_expr) =
                    ctx.select_method_by_result_type(class_name, name, def_ref.def_id, def_ref.span)
                {
                    return Ok(method_expr);
                }
                // Inside an instance-method body, a method of the SAME class
                // refers to the enclosing instance (`getB f = Box mkSt (f 1)`
                // within `instance Mk S`: `mkSt` is `mkSt @S`). The recorded
                // occurrence type may keep the class param generic there, so
                // resolve at the enclosing instance type directly.
                if let Some(inst_ty) = ctx.current_instance_type().cloned() {
                    if ctx.current_instance_class() == Some(class_name) {
                        if let Some(method_expr) = ctx.resolve_method_at_concrete_type(
                            name,
                            class_name,
                            &inst_ty,
                            def_ref.span,
                        ) {
                            return Ok(method_expr);
                        }
                    }
                }
                // Still no dict — don't try to resolve here.
                // The App case in lower_app will handle resolution
                // when the argument type is known.
                let var = ctx.lookup_var(def_ref.def_id).cloned().unwrap();
                return Ok(core::Expr::Var(var, def_ref.span));
            }

            // A bare monad-family method used as a *value* inside a point-free
            // instance method body — parsec's `instance Alternative (ParsecT …)
            // where (<|>) = mplus`, where `mplus` has no argument to drive
            // resolution. Resolve it to the enclosing instance's type's method
            // (`$instance_mplus_ParsecT`), instead of leaving it as a builtin
            // that stubs for a user type. The instance type may carry abstract
            // parameters (`ParsecT s u m`) — its head constructor is enough.
            // A monad-family dictionary that is not for THIS monad must not
            // suppress the concrete-instance dispatch; see `in_scope_dict_matches`.
            if ctx.is_monad_family_class(class_name)
                && (ctx.lookup_dict(class_name).is_none()
                    || !in_scope_dict_matches(ctx, class_name, def_ref.span))
            {
                if let Some(inst_ty) = ctx.current_instance_type().cloned() {
                    if !LowerContext::is_builtin_monad_type(&inst_ty) {
                        if let Some(method_expr) = ctx.resolve_method_at_concrete_type(
                            name,
                            class_name,
                            &inst_ty,
                            def_ref.span,
                        ) {
                            return Ok(method_expr);
                        }
                    }
                }

                // Outside an instance body: a bare monad-family method used as
                // a VALUE — `foldr (<|>) mzero ps` (parsec's `choice`). Recover
                // the monad constructor from this occurrence's typeck-recorded
                // type: the method type's first parameter (or its result, for a
                // nullary method like `mzero`) is `m a`, and `m`'s head
                // constructor is concrete for transformer types (`ParsecT s u
                // m`) even when its arguments are still variables. Dispatch to
                // the named instance method at that head; builtin monads and
                // unresolved heads fall through to the codegen fast path as
                // before.
                if let Some(occ_ty) = ctx.resolved_expr_ty_opt(def_ref.span) {
                    // Semigroup/Monoid parameterize over the VALUE type itself:
                    // take the first parameter (or result) directly, unwrapping
                    // one list for `mconcat :: [a] -> a`.
                    let is_value_class = matches!(class_name.as_str(), "Semigroup" | "Monoid");
                    let m_head = if is_value_class {
                        let target = match &occ_ty {
                            Ty::Fun(a, _) => a.as_ref(),
                            t => t,
                        };
                        Some(match target {
                            Ty::List(t) => t.as_ref().clone(),
                            // Alias-qualified container cons (`Set.Set
                            // Extension` under `import qualified Data.Set as
                            // Set`) must match the builtin instance heads,
                            // which are registered under the bare names.
                            t => strip_container_qualifier(t),
                        })
                    } else {
                        monad_head_of_method_occurrence(&occ_ty)
                    };
                    if let Some(m_head) = m_head {
                        if has_concrete_head(&m_head)
                            && !LowerContext::is_builtin_monad_type(&m_head)
                        {
                            if let Some(method_expr) = ctx.resolve_method_at_concrete_type(
                                name,
                                class_name,
                                &m_head,
                                def_ref.span,
                            ) {
                                return Ok(method_expr);
                            }
                            // `mconcat` used as a VALUE with no dedicated
                            // instance method (`mconcat <$> manyTill block eof`
                            // in pandoc's parseBlocks): its class default is
                            // `foldr mappend mempty`.
                            if name.as_str() == "mconcat" {
                                let mappend_e = ctx.resolve_method_at_concrete_type(
                                    Symbol::intern("mappend"),
                                    class_name,
                                    &m_head,
                                    def_ref.span,
                                );
                                let mempty_e = ctx.resolve_method_at_concrete_type(
                                    Symbol::intern("mempty"),
                                    class_name,
                                    &m_head,
                                    def_ref.span,
                                );
                                if let (Some(mappend_e), Some(mempty_e)) = (mappend_e, mempty_e) {
                                    let foldr_var = Var {
                                        name: Symbol::intern("foldr"),
                                        id: VarId::new(0),
                                        ty: Ty::Error,
                                    };
                                    return Ok(core::Expr::App(
                                        Box::new(core::Expr::App(
                                            Box::new(core::Expr::Var(foldr_var, def_ref.span)),
                                            Box::new(mappend_e),
                                            def_ref.span,
                                        )),
                                        Box::new(mempty_e),
                                        def_ref.span,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            // Builtin class method with no dict — fall through to regular handling
        }
    }

    // Regular variable handling
    let base_expr = if let Some(var) = ctx.lookup_var(def_ref.def_id) {
        core::Expr::Var(var.clone(), def_ref.span)
    } else {
        // Variable not found - this could be a builtin or external reference
        // Create a placeholder variable
        let placeholder = Var {
            name: Symbol::intern("unknown"),
            id: VarId::new(def_ref.def_id.index()),
            ty: Ty::Error,
        };
        core::Expr::Var(placeholder, def_ref.span)
    };

    // Check if the referenced function has user-defined class constraints
    // that need dictionary arguments.
    // IMPORTANT: Only apply dictionary-passing for USER-DEFINED classes.
    // Builtin classes (Eq, Ord, Num, Show, etc.) are handled by codegen's
    // hardcoded dispatch and must NOT go through dictionary construction,
    // because the builtin class registry uses DefIds that don't match the
    // lowering context's actual DefId assignments.
    if let Some(scheme) = ctx.lookup_scheme(def_ref.def_id) {
        let scheme_ty = scheme.ty.clone();
        // Filter to only user-defined class constraints
        let user_constraints: Vec<_> = scheme
            .constraints
            .iter()
            .filter(|c| ctx.constraint_is_dict_passed(c))
            .cloned()
            .collect();

        if !user_constraints.is_empty() {
            // Check if ALL user-class constraints have type variables
            // (meaning they can't be resolved yet — defer to App-level)
            // EXCEPTION: if a dictionary is in scope (from existential pattern match),
            // we can resolve even with type variables.
            let all_deferred = user_constraints
                .iter()
                .all(|c| c.args.iter().any(has_type_variables));
            let any_in_scope = user_constraints.iter().any(|c| {
                ctx.dict_in_scope_or_via_superclass(c.class, def_ref.span)
                    .is_some()
            });
            // One in-scope dictionary used to divert this reference into the
            // per-constraint loop below, which SKIPS any constraint it cannot
            // place — so `anyChar :: (Monad m, Stream s m Char,
            // UpdateSourcePos s Char) => …` got its `Monad` from the enclosing
            // `PandocMonad` superclass and neither of the other two, and the
            // parser was applied to its arguments one slot over. Resolving the
            // whole set at the occurrence is all-or-nothing by construction, so
            // try it first whichever dictionaries happen to be in scope.
            if let Some(dicted) =
                lower_constrained_value_all_dicts(ctx, def_ref, &scheme_ty, &user_constraints)
            {
                return Ok(dicted);
            }
            if all_deferred && !any_in_scope {
                // "Defer to App-level" assumes this reference is the head of an
                // application, so an argument type will pin the constraint. A
                // constrained *value* has no arguments and is never that head:
                // `f2 :: Mk a => a` in `print (f2 :: Int)` is an argument to
                // `print`, so nothing downstream ever resolves it and the bare
                // `\$dMk -> …` closure travels on AS the value — printing a
                // pointer, silently, with no warning.
                //
                // Typeck did record this occurrence's instantiated type. Match
                // the scheme against it to pin the constraint. Restricted to
                // non-function instantiations so genuine constrained functions
                // still defer: `lower_app` builds its own head var precisely to
                // avoid resolving a dictionary twice, and applying one here as
                // well would shift every later argument.
                // For a constrained VALUE the argument can only
                // be supplied here, and leaving it off does not leave the
                // function unsaturated — it shifts every later argument into
                // the dictionary's place. parsec's `getState :: Monad m =>
                // ParsecT s u m u` inside a class method whose OWN constraint
                // is where GHC finds the `Monad m` (`getOption :: Stream s m t
                // => …` in pandoc's `HasReaderOptions`) has no dictionary in
                // scope at all, and it was the continuation that landed in the
                // slot. A null placeholder keeps the arity honest: these
                // dictionaries are threaded to a callee that never selects
                // through them, and a crash reading one is a better outcome
                // than a parser jumping through its own continuation.
                if !matches!(
                    occurrence_or_scheme_ty(ctx, def_ref, &scheme_ty),
                    Ty::Fun(..)
                ) {
                    let mut result = base_expr;
                    for _ in &user_constraints {
                        result = core::Expr::App(
                            Box::new(result),
                            Box::new(core::Expr::Lit(
                                core::Literal::Int(0),
                                Ty::Error,
                                def_ref.span,
                            )),
                            def_ref.span,
                        );
                    }
                    return Ok(result);
                }
                return Ok(base_expr);
            }

            // The occurrence's instantiation, used to pin constraints the
            // declared scheme leaves open: `char :: (Monad m, Stream s m Char,
            // UpdateSourcePos s Char) => Char -> ParsecT s u m Char` used at
            // `Sources` pins `s` even where `m` stays a variable.
            let occ_subst = refined_occurrence_ty(ctx, def_ref.span).map(|occ| {
                let mut subst = bhc_types::Subst::new();
                match_ty(&scheme_ty, &occ, &mut subst);
                subst
            });

            // Apply dictionaries for each user-defined class constraint
            let mut result = base_expr;
            for constraint in &user_constraints {
                // Skip constraints with type variables UNLESS a dict is in
                // scope — directly, or as a superclass of one that is.
                let from_scope =
                    ctx.dict_in_scope_or_via_superclass(constraint.class, def_ref.span);
                let instantiated = occ_subst.as_ref().map(|subst| {
                    Constraint::new_multi(
                        constraint.class,
                        constraint.args.iter().map(|a| subst.apply(a)).collect(),
                        constraint.span,
                    )
                });
                // Resolvable as soon as the occurrence pins ONE argument:
                // `instance Monad m => Stream Sources m Char` is selected by
                // `Sources` alone, and requiring every argument to be concrete
                // rejected it for the `m` a constrained binding still holds
                // open.
                let from_occurrence = instantiated
                    .as_ref()
                    .filter(|c| c.args.iter().any(|a| !has_type_variables(a)))
                    .and_then(|c| ctx.resolve_dictionary(c, def_ref.span));
                if constraint.args.iter().any(has_type_variables)
                    && from_scope.is_none()
                    && from_occurrence.is_none()
                {
                    // Nothing can fill this one. Skipping it does not leave the
                    // reference unsaturated — it moves every later dictionary,
                    // and then every value argument, one slot up. `anyChar`'s
                    // `Stream`/`UpdateSourcePos` were skipped this way while its
                    // `Monad` came from the enclosing superclass, and pandoc's
                    // `string = mapM char` applied a parser to its continuation.
                    result = core::Expr::App(
                        Box::new(result),
                        Box::new(core::Expr::Lit(
                            core::Literal::Int(0),
                            Ty::Error,
                            def_ref.span,
                        )),
                        def_ref.span,
                    );
                    continue;
                }

                // Try to resolve the dictionary
                if let Some(dict_expr) = from_occurrence
                    .or_else(|| ctx.resolve_dictionary(constraint, def_ref.span))
                    .or(from_scope)
                {
                    result = core::Expr::App(Box::new(result), Box::new(dict_expr), def_ref.span);
                } else {
                    // Dictionary not available - generate an error expression
                    let error_msg = format!(
                        "No {} dictionary available for constraint {}",
                        constraint.class.as_str(),
                        format_constraint(constraint)
                    );
                    let error_expr = make_error_expr(&error_msg, def_ref.span);
                    result = core::Expr::App(Box::new(result), Box::new(error_expr), def_ref.span);
                }
            }
            return Ok(result);
        }
    }

    Ok(base_expr)
}

/// Every leaf of the type is a concrete constructor/primitive — no type
/// variables, no `Ty::Error`. Downstream dictionary resolution can act on
/// such a type without guessing.
fn is_fully_concrete_ty(ty: &Ty) -> bool {
    match ty {
        Ty::Con(_) | Ty::Prim(_) => true,
        Ty::App(f, a) => is_fully_concrete_ty(f) && is_fully_concrete_ty(a),
        Ty::Fun(a, b) => is_fully_concrete_ty(a) && is_fully_concrete_ty(b),
        Ty::List(t) => is_fully_concrete_ty(t),
        Ty::Tuple(ts) => ts.iter().all(is_fully_concrete_ty),
        _ => false,
    }
}

/// Try to infer the concrete type of an HIR expression.
///
/// Returns `Some(Ty)` for expressions with obvious types:
/// - Constructors: look up the constructor's type name
/// - Int/Float/Char/String literals: return the corresponding type
/// - Other expressions: return None (type not inferrable without type checker)
fn try_infer_arg_type(ctx: &LowerContext, expr: &hir::Expr) -> Option<Ty> {
    match expr {
        Expr::Con(def_ref) => {
            // Look up the constructor's data type
            ctx.lookup_constructor(def_ref.def_id)
                .map(|con_info| Ty::Con(TyCon::new(con_info.type_name, Kind::Star)))
        }
        Expr::Var(def_ref) => {
            // Look up the type of this variable from the type checker.
            // Only return concrete (non-polymorphic) types. FULLY-concrete
            // applied types count (`p :: ParsecT [Char] () Identity Char`
            // from an enclosing signature) — they carry everything downstream
            // dictionary resolution needs; bare heads or var-carrying types
            // do not and must stay None.
            //
            // Expand type synonyms on the way out. The type recorded for a
            // parameter is whatever its signature said, so a parameter
            // written `String` stays `Con "String"` and cannot match the
            // instance head `Stream [tok] m tok`. A constrained value in that
            // scope — parsec's `anyChar`, nullary and needing only a `Stream`
            // dictionary — then resolves to no instance and is emitted with
            // its dictionary MISSING, silently and with no unresolved-dictionary
            // warning; consumers read that undicted closure's header as the
            // parser's own. Spelling the same signature `[Char]` was the
            // difference between a working parser and a crash.
            //
            // The expansion happens HERE, on the type handed to instance
            // resolution, rather than on the parameter's recorded type: giving
            // codegen the expanded form makes it infer a different width for
            // Char and emit `icmp eq i32 %to_char, i64 32`.
            let ty = ctx.expand_type_aliases(&ctx.lookup_type(def_ref.def_id));
            let allow_concrete = std::env::var("BHC_NO_CONCVAR").is_err();
            match &ty {
                Ty::Con(_) => Some(ty),
                t if allow_concrete && is_fully_concrete_ty(t) => Some(ty),
                Ty::Error => {
                    // Core IR params often have ty: Error in type_schemes.
                    // Check the Core Var's type as a fallback (populated from
                    // the function's type signature in compile_equations).
                    if let Some(var) = ctx.lookup_var(def_ref.def_id) {
                        let vty = ctx.expand_type_aliases(&var.ty);
                        match &vty {
                            Ty::Con(_) => Some(vty),
                            t if allow_concrete && is_fully_concrete_ty(t) => Some(vty),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        Expr::Lit(lit, _) => match lit {
            Lit::Int(_) => Some(Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star))),
            Lit::Float(_) => Some(Ty::Con(TyCon::new(Symbol::intern("Double"), Kind::Star))),
            Lit::Char(_) => Some(Ty::Con(TyCon::new(Symbol::intern("Char"), Kind::Star))),
            Lit::String(_) => Some(Ty::List(Box::new(Ty::Con(TyCon::new(
                Symbol::intern("Char"),
                Kind::Star,
            ))))),
        },
        // List literal `[a, b, ...]`: infer the element type from the first
        // element, yielding `[elem]`. Enables resolving dictionaries for
        // functions like `f :: C a => [a] -> r` applied to a list literal.
        Expr::List(elems, _) => {
            let elem_ty = try_infer_arg_type(ctx, elems.first()?)?;
            Some(Ty::List(Box::new(elem_ty)))
        }
        Expr::App(f, x, _) => {
            // Peel off App layers to find head constructor and collect args:
            // App(App(Con(Pair), x), y) → (Con(Pair), [x, y])
            let mut head = f.as_ref();
            let mut con_args = vec![x.as_ref()];
            while let Expr::App(inner_f, inner_x, _) = head {
                con_args.push(inner_x.as_ref());
                head = inner_f.as_ref();
            }
            con_args.reverse();
            if let Expr::Con(def_ref) = head {
                if let Some(con_info) = ctx.lookup_constructor(def_ref.def_id) {
                    let base = Ty::Con(TyCon::new(con_info.type_name, Kind::Star));
                    // Try to determine the result type using the constructor's type
                    // signature. If the type checker has the constructor's type, use it.
                    let con_ty = ctx.lookup_type(def_ref.def_id);
                    if let Some(result_ty) =
                        extract_constructor_result_type(&con_ty, &con_args, ctx)
                    {
                        return Some(result_ty);
                    }
                    // Return bare type name. The caller (lower_app) will try
                    // applied-type resolution as a second attempt if this fails.
                    return Some(base);
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract the result type from a constructor's type signature, substituting
/// type variables based on inferred argument types.
///
/// For example, given `MkPair :: a -> a -> Pair a` and args `[Red, Blue]`:
/// 1. Peel Fun layers: `a -> a -> Pair a` → params `[a, a]`, result `Pair a`
/// 2. Infer arg types: `Red :: Color`, `Blue :: Color`
/// 3. Build substitution: `{a -> Color}`
/// 4. Apply to result: `Pair Color` = `App(Con("Pair"), Con("Color"))`
fn extract_constructor_result_type(
    con_ty: &Ty,
    con_args: &[&hir::Expr],
    ctx: &LowerContext,
) -> Option<Ty> {
    // Peel off Fun layers to get parameter types and result type
    let mut param_tys = Vec::new();
    let mut current = con_ty;
    while let Ty::Fun(arg, ret) = current {
        param_tys.push(arg.as_ref().clone());
        current = ret;
    }

    // If the constructor type is Error or we have no params, fall back
    if matches!(current, Ty::Error) || param_tys.is_empty() {
        return None;
    }

    let result_ty = current;

    // Build a substitution by matching param types against inferred arg types
    let mut subst = bhc_types::Subst::new();
    for (param_ty, arg_expr) in param_tys.iter().zip(con_args.iter()) {
        if let Ty::Var(tv) = param_ty {
            if subst.get(tv).is_none() {
                if let Some(arg_ty) = try_infer_arg_type(ctx, arg_expr) {
                    subst.insert(tv, arg_ty);
                }
            }
        }
    }

    // Apply substitution to the result type
    let concrete_result = subst.apply(result_ty);
    // WIP lowering: the branch condition is retained intentionally as a marker
    // for where the "more specific than the original" check will diverge once
    // the fallback path is implemented; both arms currently agree.
    #[allow(clippy::if_same_then_else)]
    if concrete_result != *result_ty || !matches!(result_ty, Ty::App(_, _)) {
        Some(concrete_result)
    } else {
        Some(concrete_result)
    }
}

/// Try to infer an applied type from a constructor application expression.
///
/// For bare constructors like `Red`, returns None (use `try_infer_arg_type`).
/// For applied constructors like `Wrap Green`:
/// - Head: Wrap → type name "Wrapper"
/// - Arg: Green → type Color
/// - Returns: `App(Con("Wrapper"), Con("Color"))`
///
/// This enables resolution of parameterized instance types like
/// `instance Describable a => Describable (Wrapper a)`.
fn try_infer_applied_type(ctx: &LowerContext, expr: &hir::Expr) -> Option<Ty> {
    let Expr::App(_, _, _) = expr else {
        return None;
    };

    let mut head = expr;
    let mut con_args = Vec::new();
    while let Expr::App(f, x, _) = head {
        con_args.push(x.as_ref());
        head = f.as_ref();
    }
    con_args.reverse();

    let def_ref = match head {
        Expr::Con(dr) => dr,
        _ => return None,
    };

    let con_info = ctx.lookup_constructor(def_ref.def_id)?;
    let base = Ty::Con(TyCon::new(con_info.type_name, Kind::Star));

    let mut arg_types = Vec::new();
    for arg in &con_args {
        if let Some(ty) = try_infer_arg_type(ctx, arg) {
            arg_types.push(ty);
        }
    }

    if arg_types.is_empty() {
        return None;
    }

    // Deduplicate: for `Pair a a`, both value args map to the same type param
    let mut unique_types = Vec::new();
    for ty in &arg_types {
        if !unique_types.contains(ty) {
            unique_types.push(ty.clone());
        }
    }

    // Build applied type: App(App(base, ty1), ty2)
    let mut result = base;
    for ty in unique_types {
        result = Ty::App(Box::new(result), Box::new(ty));
    }

    Some(result)
}

/// Peel an application chain to find the head variable and collected arguments.
///
/// Given `App(App(Var(f), a1), a2)`, returns `Some((f_def_ref, [a1, a2]))`.
/// Arguments are returned in application order (inside-out).
fn peel_app_chain(expr: &hir::Expr) -> Option<(&DefRef, Vec<&hir::Expr>)> {
    let mut args = Vec::new();
    let mut current = expr;

    // Walk the App chain collecting arguments
    while let Expr::App(f, x, _) = current {
        args.push(x.as_ref());
        current = f.as_ref();
    }

    // The head must be a Var
    if let Expr::Var(def_ref) = current {
        // Reverse so args are in application order (innermost first)
        args.reverse();
        Some((def_ref, args))
    } else {
        None
    }
}

/// Structurally match a (possibly polymorphic) parameter type against an
/// inferred concrete type, recording type-variable bindings in `subst`.
/// Only the first binding for each variable is kept.
pub(crate) fn match_ty(param: &Ty, concrete: &Ty, subst: &mut bhc_types::Subst) {
    match (param, concrete) {
        (Ty::Var(tv), c) => {
            // Concrete wins over an earlier var-to-var binding: typeck can
            // record an occurrence whose PARAMETER portion keeps signature
            // variables while its RESULT portion is substituted
            // (`optional`'s occ = `ParsecT s u m Char -> ParsecT [Char] ()
            // m' a'`); first-wins bound `s := s-var` from the parameter side
            // and the concrete `[Char]` from the result side was discarded —
            // the Stream dictionary then failed to resolve. No cloning in the
            // check: this is a hot path (a stray `prev.clone()` here made
            // citeproc-heavy modules ~50x slower to compile).
            let upgrade = match subst.get(tv) {
                None => true,
                Some(prev) => has_type_variables(prev) && !has_type_variables(c),
            };
            if upgrade {
                subst.insert(tv, c.clone());
            }
        }
        (Ty::List(p), Ty::List(c)) => match_ty(p, c, subst),
        (Ty::App(p1, p2), Ty::App(c1, c2)) => {
            match_ty(p1, c1, subst);
            match_ty(p2, c2, subst);
        }
        (Ty::Fun(p1, p2), Ty::Fun(c1, c2)) => {
            match_ty(p1, c1, subst);
            match_ty(p2, c2, subst);
        }
        (Ty::Tuple(ps), Ty::Tuple(cs)) if ps.len() == cs.len() => {
            for (p, c) in ps.iter().zip(cs) {
                match_ty(p, c, subst);
            }
        }
        _ => {}
    }
}

/// For a constrained user *function* (not a class method) applied to `args`,
/// resolve the dictionary argument(s) it expects. The constrained type
/// variables are instantiated by matching the function's declared parameter
/// types against the inferred argument types, then a dictionary is resolved
/// per constraint at those types (concrete instance, or an in-scope dictionary
/// for the deferred/recursive case). Returns the dictionaries in constraint
/// order, or `None` if the head is not user-constrained or any dictionary
/// cannot be resolved (the caller then falls back to plain lowering).
/// A constrained callee's declared parameter types and a substitution that
/// instantiates its type variables from the inferable argument types. Shared by
/// dictionary resolution and by argument lowering, so a constrained *value*
/// argument can have its OWN dictionary applied at the resulting concrete type.
/// The FULLY-CONCRETE result type of an applied expression, computed from
/// the head's scheme with its parameters pinned by the (inferable) argument
/// types. `manyTill p $eta` inside a signed binding pins `s`, `u`, `m` from
/// the params' types and yields `ParsecT [Char] () Identity [Char]` — enough
/// for method dispatch AND operand dictionary resolution. A bare `digit`
/// operand pins nothing (its scheme keeps variables) and stays None, so the
/// validated codegen fast paths keep those cases.
fn concrete_applied_result_ty(ctx: &LowerContext, e: &hir::Expr) -> Option<Ty> {
    let mut head = e;
    let mut args: Vec<&hir::Expr> = Vec::new();
    while let hir::Expr::App(f, a, _) = head {
        args.push(a.as_ref());
        head = f.as_ref();
    }
    if args.is_empty() {
        return None;
    }
    args.reverse();
    let hir::Expr::Var(head_ref) = head else {
        return None;
    };
    let dbg = std::env::var("BHC_DBG_CART").is_ok();
    let Some(sch) = ctx.lookup_scheme(head_ref.def_id) else {
        if dbg {
            eprintln!(
                "[cart] head {:?} ({:?}): NO scheme",
                ctx.lookup_var(head_ref.def_id).map(|v| v.name),
                head_ref.def_id
            );
        }
        return None;
    };
    let scheme_ty = sch.ty.clone();
    if dbg {
        eprintln!(
            "[cart] head {:?}: scheme {:?}",
            ctx.lookup_var(head_ref.def_id).map(|v| v.name),
            scheme_ty
        );
    }
    let mut param_tys = Vec::new();
    let mut cur = &scheme_ty;
    while let Ty::Fun(a, r) = cur {
        param_tys.push(a.as_ref().clone());
        cur = r.as_ref();
    }
    if args.len() > param_tys.len() {
        return None;
    }
    let mut subst = bhc_types::Subst::new();
    for (p, a) in param_tys.iter().zip(args.iter()) {
        if let Some(at) = try_infer_arg_type(ctx, a) {
            match_ty(p, &at, &mut subst);
        }
    }
    let mut result = scheme_ty.clone();
    for _ in 0..args.len() {
        let Ty::Fun(_, r) = result else { return None };
        result = *r;
    }
    let result = subst.apply(&result);
    // The MONAD portion must be concrete — the head constructor and every
    // applied argument except the LAST (the value type: `ParsecT [Char] ()
    // Identity [a]`'s `a` is irrelevant to instance resolution and operand
    // dictionaries).
    fn monad_applied_concrete(t: &Ty) -> bool {
        let Ty::App(f, _last) = t else { return false };
        let mut cur = f.as_ref();
        loop {
            match cur {
                Ty::App(g, a) => {
                    if !is_fully_concrete_ty(a) {
                        return false;
                    }
                    cur = g.as_ref();
                }
                Ty::Con(_) => return true,
                _ => return false,
            }
        }
    }
    let ok = monad_applied_concrete(&result);
    if dbg {
        eprintln!("[cart] result {:?} monad_concrete={}", result, ok);
    }
    ok.then_some(result)
}

/// Apply dictionaries to a reference to a constrained *value* — a binding whose
/// scheme carries constraints but whose instantiated type is not a function, so
/// it is never the head of an application and no argument will ever drive
/// resolution (`f2 :: Mk a => a`, `mkVal :: Tagged m => m Int`).
///
/// Returns `None` — leaving the caller to emit the bare variable, as before —
/// unless the occurrence's recorded type pins every constraint to a fully
/// concrete one *and* every dictionary resolves. Half-applying dictionaries
/// would shift argument slots, which is worse than not applying them.
/// The type this occurrence was inferred at, or the declared one when typeck
/// recorded nothing for the span. Used only to tell a constrained VALUE from a
/// constrained FUNCTION.
fn occurrence_or_scheme_ty(ctx: &LowerContext, def_ref: &DefRef, scheme_ty: &Ty) -> Ty {
    ctx.resolved_expr_ty_opt(def_ref.span)
        .unwrap_or_else(|| scheme_ty.clone())
}

/// Every dictionary a constrained VALUE needs, from whichever source has it.
///
/// Resolving the whole set at once, and only when the occurrence pins EVERY
/// constraint to a concrete type, was not enough. `anyChar :: (Monad m, Stream s m Char,
/// UpdateSourcePos s Char) => ParsecT s u m Char` used inside a
/// `PandocMonad m =>` binding pins `s` and `u` but leaves `m` a variable, so it
/// gave up — and the per-constraint loop then applied only the `Monad` it could
/// reach through the enclosing superclass and SKIPPED the other two, so the
/// parser was applied to its arguments two slots over.
///
/// Each constraint is resolved at the occurrence-instantiated type, else from a
/// dictionary in scope (directly or as a superclass), else filled with a null
/// placeholder. One argument per constraint, always: a missing slot does not
/// leave the value unsaturated, it shifts everything after it.
fn lower_constrained_value_all_dicts(
    ctx: &mut LowerContext,
    def_ref: &DefRef,
    scheme_ty: &Ty,
    constraints: &[Constraint],
) -> Option<core::Expr> {
    let occ_ty = refined_occurrence_ty(ctx, def_ref.span)?;
    let mut subst = bhc_types::Subst::new();
    match_ty(scheme_ty, &occ_ty, &mut subst);
    // A function instantiation means this is a constrained *function*; those
    // resolve at the application site, which builds its own head var.
    if matches!(subst.apply(scheme_ty), Ty::Fun(..)) {
        return None;
    }
    let var = ctx.lookup_var(def_ref.def_id).cloned()?;
    let mut dicts = Vec::with_capacity(constraints.len());
    let mut any_real = false;
    for c in constraints {
        let instantiated = Constraint::new_multi(
            c.class,
            c.args.iter().map(|a| subst.apply(a)).collect(),
            c.span,
        );
        let dict = ctx
            .resolve_dictionary(&instantiated, def_ref.span)
            .or_else(|| ctx.dict_in_scope_or_via_superclass(c.class, def_ref.span));
        if dict.is_some() {
            any_real = true;
        }
        dicts.push(dict.unwrap_or(core::Expr::Lit(
            core::Literal::Int(0),
            Ty::Error,
            def_ref.span,
        )));
    }
    // Nothing was found at all — leave the reference alone rather than burying
    // it under placeholders; the caller's own paths may still do better.
    if !any_real {
        return None;
    }
    let mut result = core::Expr::Var(var, def_ref.span);
    for dict in dicts {
        result = core::Expr::App(Box::new(result), Box::new(dict), def_ref.span);
    }
    Some(result)
}

fn callee_param_tys_and_subst(
    ctx: &mut LowerContext,
    head_ref: &DefRef,
    args: &[&hir::Expr],
) -> Option<(Vec<Ty>, bhc_types::Subst)> {
    let scheme_ty = ctx.lookup_scheme(head_ref.def_id)?.ty.clone();
    let mut param_tys = Vec::new();
    let mut cur = &scheme_ty;
    while let Ty::Fun(a, r) = cur {
        param_tys.push(a.as_ref().clone());
        cur = r.as_ref();
    }
    let mut subst = bhc_types::Subst::new();
    for (p, arg) in param_tys.iter().zip(args.iter()) {
        if let Some(at) = try_infer_arg_type(ctx, arg) {
            match_ty(p, &at, &mut subst);
        }
    }
    // Arguments alone may not pin every scheme variable — inside a top-level
    // CAF (`p1 = choice [char 'y', char 'x'] :: Parser Char`) no sibling
    // argument carries the stream type `s`. Typeck recorded this occurrence's
    // instantiated type (concrete via the binding's annotation); match the
    // scheme against it to fill the remaining variables (argument-driven
    // bindings win — `match_ty` never overrides).
    if let Some(occ_ty) = ctx
        .resolved_expr_ty_opt(head_ref.span)
        .or_else(|| ctx.expr_ty_opt(head_ref.span))
    {
        match_ty(&scheme_ty, &occ_ty, &mut subst);
    }
    Some((param_tys, subst))
}

/// Whether the type contains an error hole. A hint built from one would bind a
/// constrained variable to nothing and make a leaf *look* resolved.
fn ty_has_error(ty: &Ty) -> bool {
    match ty {
        Ty::Error => true,
        Ty::Var(_) | Ty::Con(_) | Ty::Prim(_) | Ty::Nat(_) | Ty::TyList(_) => false,
        Ty::App(f, a) | Ty::Fun(f, a) => ty_has_error(f) || ty_has_error(a),
        Ty::Tuple(tys) => tys.iter().any(ty_has_error),
        Ty::List(elem) => ty_has_error(elem),
        Ty::Forall(_, body) => ty_has_error(body),
    }
}

/// A compact one-line rendering of a type, for the propagation trace.
pub(crate) fn dbg_ty(ty: &Ty) -> String {
    match ty {
        Ty::Var(v) => format!("t{}", v.id),
        Ty::Con(c) => c.name.as_str().to_string(),
        Ty::Prim(_) => "#".into(),
        Ty::App(f, a) => format!("({} {})", dbg_ty(f), dbg_ty(a)),
        Ty::Fun(a, r) => format!("({} -> {})", dbg_ty(a), dbg_ty(r)),
        Ty::Tuple(ts) => format!("({})", ts.iter().map(dbg_ty).collect::<Vec<_>>().join(",")),
        Ty::List(e) => format!("[{}]", dbg_ty(e)),
        Ty::Forall(_, b) => dbg_ty(b),
        Ty::Error => "ERR".into(),
        Ty::Nat(_) | Ty::TyList(_) => "?".into(),
    }
}

/// How many arrows a type has at its top level: `a -> b -> c` has two.
fn arrow_count(ty: &Ty) -> usize {
    let mut n = 0;
    let mut cur = ty;
    while let Ty::Fun(_, r) = cur {
        n += 1;
        cur = r.as_ref();
    }
    n
}

/// Push a caller's expected type inward, recording it for every sub-expression
/// that shares it, so a constrained value buried under an unconstrained head
/// can still resolve its own dictionary.
///
/// `readWithM (try (id pB)) …` is the shape that needs this: `readWithM`'s
/// parameter names the stream type `Sources`, `try` and `id` pass it straight
/// through, and `pB :: Stream s m Char => …` is the leaf whose dictionary
/// depends on it. Typeck recorded `pB`'s occurrence as a row of bare variables,
/// so without the hint the constraint resolves to nothing and the slot becomes
/// a null placeholder — which is a parser run through its own continuation.
/// The head's type comes from typeck rather than from a scheme, so a BUILTIN
/// head (`maybe`, `id`, `either`) carries the type inward just as a declared
/// one does; this is why pandoc's `orderedListStart` (`try (maybe
/// anyOrderedListMarker … mbstydelim)`) is reached at all.
fn prop_dbg() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var_os("BHC_DBG_PROP").is_some())
}

pub(crate) fn propagate_expected_ty(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    expected: &Ty,
    depth: u32,
) {
    if prop_dbg() {
        let kind = match expr {
            Expr::Var(dr) => format!(
                "Var {:?}",
                ctx.lookup_var(dr.def_id)
                    .map(|v| v.name.as_str().to_string())
            ),
            Expr::App(..) => "App".into(),
            Expr::Lam(..) => "Lam".into(),
            Expr::Let(..) => "Let".into(),
            Expr::Case(..) => "Case".into(),
            other => format!("{:?}", std::mem::discriminant(other)),
        };
        eprintln!(
            "{}PROP {} <- {}",
            "  ".repeat(depth as usize),
            kind,
            crate::expr::dbg_ty(expected)
        );
    }
    // The walk is bounded: a deeply nested spine gains nothing from more
    // levels, and the hint only ever refines what typeck already recorded.
    if depth > 6 || ty_has_error(expected) {
        return;
    }
    match expr {
        Expr::Var(def_ref) => {
            ctx.record_expected_ty(def_ref.span, expected.clone());
        }
        Expr::Ann(inner, _, _) | Expr::TypeApp(inner, _, _) => {
            propagate_expected_ty(ctx, inner, expected, depth + 1);
        }
        Expr::If(_, then_e, else_e, _) => {
            propagate_expected_ty(ctx, then_e, expected, depth + 1);
            propagate_expected_ty(ctx, else_e, expected, depth + 1);
        }
        Expr::Let(_, body, _) => {
            propagate_expected_ty(ctx, body, expected, depth + 1);
        }
        Expr::Case(_, alts, _) => {
            for alt in alts {
                propagate_expected_ty(ctx, &alt.rhs, expected, depth + 1);
            }
        }
        Expr::Lam(pats, body, _) => {
            let mut result = expected;
            for _ in 0..pats.len() {
                let Ty::Fun(_, ret) = result else { return };
                result = ret.as_ref();
            }
            let result = result.clone();
            propagate_expected_ty(ctx, body, &result, depth + 1);
        }
        Expr::App(..) => {
            let mut head: &hir::Expr = expr;
            let mut args: Vec<&hir::Expr> = Vec::new();
            while let Expr::App(f, a, _) = head {
                args.push(a.as_ref());
                head = f.as_ref();
            }
            args.reverse();
            while let Expr::TypeApp(inner, _, _) | Expr::Ann(inner, _, _) = head {
                head = inner.as_ref();
            }
            // A monadic operator's own type does not carry the hint: typeck
            // records `>>=` at its occurrence as `(a -> m b) -> m b` — one
            // arrow for two arguments — so the spine walk below bails and the
            // whole right-hand side of every `do` block loses the stream type.
            // The shape is known instead: both operands live in the SAME
            // monad as the result, differing only in the value they carry.
            if let Expr::Var(dr) = head {
                let name = ctx.lookup_var(dr.def_id).map(|v| v.name);
                if let Some(operand_tys) =
                    name.and_then(|n| monadic_operand_tys(n.as_str(), expected, args.len()))
                {
                    for (want, a) in operand_tys.iter().zip(args.iter()) {
                        if let Some(want) = want {
                            propagate_expected_ty(ctx, a, want, depth + 1);
                        }
                    }
                    return;
                }
            }
            // `f $ x` IS `f x` for this purpose. Peeling `$`'s own scheme
            // stops at its result variable and carries nothing inward, and
            // pandoc writes every one of these parsers as `try $ do …`.
            if let Expr::Var(dr) = head {
                if ctx
                    .lookup_var(dr.def_id)
                    .is_some_and(|v| v.name.as_str() == "$")
                    && args.len() >= 2
                {
                    let rebuilt_head = args[0];
                    let rest: Vec<&hir::Expr> = args[1..].to_vec();
                    propagate_spine(ctx, rebuilt_head, &rest, expected, depth);
                    return;
                }
            }
            propagate_spine(ctx, head, &args, expected, depth);
        }
        _ => {}
    }
}

/// The expected type of each operand of a monadic/applicative operator, given
/// the expected type of the whole application. `None` for an operand whose type
/// this says nothing about.
///
/// The monad is whatever `expected` is applied to — `ParsecT Sources ParserState
/// m` for `ParsecT Sources ParserState m Int` — and the carried value is left
/// unknown, which is all a nested parser needs to resolve its `Stream`
/// dictionary.
fn monadic_operand_tys(name: &str, expected: &Ty, nargs: usize) -> Option<Vec<Option<Ty>>> {
    if nargs != 2 {
        return None;
    }
    let Ty::App(monad, _) = expected else {
        return None;
    };
    // A stand-in for "some value type" — deliberately outside the ranges
    // typeck and lowering allocate from, so it can never capture a real one.
    let unknown = Ty::Var(bhc_types::TyVar::new_star(0xFFE0_0000));
    let in_monad = Ty::App(monad.clone(), Box::new(unknown));
    let whole = Some(expected.clone());
    let operand = Some(in_monad);
    match name {
        // Both operands are actions; the result is the second one's.
        ">>" | "*>" => Some(vec![operand, whole]),
        // The first operand's action IS the result; the second is discarded.
        "<*" => Some(vec![whole, operand]),
        // `p >>= k`: `p` is an action, `k` returns the result.
        ">>=" => Some(vec![operand, None]),
        "=<<" => Some(vec![None, operand]),
        // Both branches have the whole type.
        "<|>" | "mplus" => Some(vec![whole.clone(), whole]),
        // `f <$> p` / `mf <*> p`: only the right operand is an action here.
        "<$>" | "fmap" | "<*>" => Some(vec![None, operand]),
        _ => None,
    }
}

/// The application-spine half of [`propagate_expected_ty`]: solve the head's
/// type variables from the expected result type (and from whatever the sibling
/// arguments already pin), then push each instantiated parameter type into the
/// matching argument.
fn propagate_spine(
    ctx: &mut LowerContext,
    head: &hir::Expr,
    args: &[&hir::Expr],
    expected: &Ty,
    depth: u32,
) {
    let mut head = head;
    while let Expr::TypeApp(inner, _, _) | Expr::Ann(inner, _, _) = head {
        head = inner.as_ref();
    }
    // The head's DECLARED type, else its instantiated one. The
    // declared type is preferred because typeck can record an
    // occurrence whose result is still an unsolved variable — `try`
    // arrives as `ParsecT ?s ?u ?m Int -> ?r`, and matching `?r`
    // against the expected type binds nothing the parameter shares.
    // The occurrence is the fallback that carries the hint through a
    // BUILTIN head (`id`, `maybe`, `either`), which has no scheme.
    let declared = match head {
        Expr::Var(dr) => ctx.lookup_scheme(dr.def_id).map(|sc| sc.ty.clone()),
        _ => None,
    };
    let head_ty = declared
        .filter(|t| arrow_count(t) >= args.len())
        .or_else(|| ctx.resolved_expr_ty_opt(head.span()));
    if prop_dbg() {
        let hn = match head {
            Expr::Var(dr) | Expr::Con(dr) => format!(
                "{:?}",
                ctx.lookup_var(dr.def_id)
                    .map(|v| v.name.as_str().to_string())
            ),
            o => format!("{:?}", std::mem::discriminant(o)),
        };
        eprintln!(
            "{}SPINE head={} nargs={} ty={}",
            "  ".repeat(depth as usize),
            hn,
            args.len(),
            head_ty.as_ref().map_or("NONE".to_string(), dbg_ty)
        );
    }
    let Some(head_ty) = head_ty else {
        return;
    };
    let mut param_tys: Vec<Ty> = Vec::new();
    let mut cur = &head_ty;
    for _ in 0..args.len() {
        let Ty::Fun(a, r) = cur else { return };
        param_tys.push(a.as_ref().clone());
        cur = r.as_ref();
    }
    let mut subst = bhc_types::Subst::new();
    match_ty(cur, expected, &mut subst);
    // Sibling arguments pin what the result type leaves open.
    for (pty, a) in param_tys.iter().zip(args.iter()) {
        if let Some(at) = try_infer_arg_type(ctx, a) {
            match_ty(pty, &at, &mut subst);
        }
    }
    for (pty, a) in param_tys.iter().zip(args.iter()) {
        let inner = subst.apply(pty);
        propagate_expected_ty(ctx, a, &inner, depth + 1);
    }
    // The head's own instantiated type, so a CLASS METHOD at this position
    // knows which monad it is being used at even where typeck recorded only a
    // variable — the whole body of an instance method is such a place.
    if let Expr::Var(dr) = head {
        let instantiated = subst.apply(&head_ty);
        if !ty_has_error(&instantiated) {
            ctx.record_expected_ty(dr.span, instantiated);
        }
    }
}

/// The type of the occurrence at `span`, refined by any expected type a caller
/// pushed inward. The recorded type keeps its structure — the hint only
/// substitutes its variables — so a hint that is *less* informative than what
/// typeck inferred cannot lose anything.
fn refined_occurrence_ty(ctx: &LowerContext, span: Span) -> Option<Ty> {
    let recorded = ctx.resolved_expr_ty_opt(span);
    let Some(hint) = ctx.expected_ty_opt(span) else {
        return recorded;
    };
    let Some(recorded) = recorded else {
        return Some(hint);
    };
    let mut subst = bhc_types::Subst::new();
    match_ty(&recorded, &hint, &mut subst);
    Some(subst.apply(&recorded))
}

/// Lower a value argument, applying its dictionaries when it is a constrained
/// function/value used at a known concrete type — e.g. a parser
/// `digit :: Stream s Identity t => Parsec s u Char` passed to `parse`, whose
/// `Stream` dictionary must be supplied so it is a usable ParsecT rather than a
/// `\$d -> ParsecT …` function (whose newtype field, read as a function
/// pointer, would be garbage). Falls back to plain lowering otherwise.
fn lower_value_arg(
    ctx: &mut LowerContext,
    arg: &hir::Expr,
    expected_ty: Option<&Ty>,
) -> LowerResult<core::Expr> {
    if let Some(exp) = expected_ty {
        // Record what this position expects for every sub-expression that
        // shares the type, so a constrained value under an unconstrained head
        // can resolve its own dictionary. See `propagate_expected_ty`.
        propagate_expected_ty(ctx, arg, exp, 0);
        // Even if the expected type still has variables (e.g. a parser's result
        // type `a`, fixed by context rather than by arguments), attempt dict
        // application: `lower_constrained_fn_value` only requires the
        // *constraint's* own type arguments to be concrete, and returns None
        // otherwise so we fall back to plain lowering.
        if is_constrained_fn_value(ctx, arg) {
            if let Some(e) = lower_constrained_fn_value(ctx, arg, exp) {
                return Ok(e);
            }
        }
        // Application-form constrained value: a parser combinator with its own
        // constraint, partially applied — `char 'a'`, `many1 digit`. The
        // dictionaries must precede the value arguments, so this cannot be left
        // to plain lowering (which would shift every argument by one).
        if let Some(e) = lower_constrained_fn_app(ctx, arg, exp)? {
            return Ok(e);
        }
        // A LIST literal of parsers — `choice [char 'a', digit]` expects
        // `[ParsecT s u m a]`. Each element is itself a value at the list's
        // element type and may be a constrained parser needing its dictionary;
        // plain `lower_list` would leave every element an undicted arity-0
        // value. Rebuild the cons spine with elements lowered as value args.
        if let (hir::Expr::List(elems, list_span), Ty::List(elem_ty)) = (arg, exp) {
            let nil_var = Var {
                name: Symbol::intern("[]"),
                id: VarId::new(0),
                ty: Ty::Error,
            };
            let cons_var = Var {
                name: Symbol::intern(":"),
                id: VarId::new(0),
                ty: Ty::Error,
            };
            let mut result = core::Expr::Var(nil_var, *list_span);
            for elem in elems.iter().rev() {
                let elem_core = lower_value_arg(ctx, elem, Some(elem_ty.as_ref()))?;
                let cons_app = core::Expr::App(
                    Box::new(core::Expr::Var(cons_var.clone(), *list_span)),
                    Box::new(elem_core),
                    *list_span,
                );
                result = core::Expr::App(Box::new(cons_app), Box::new(result), *list_span);
            }
            return Ok(result);
        }
    }
    lower_expr(ctx, arg)
}

/// `liftM`/`liftA` are the `Monad`/`Applicative`-constrained spellings of
/// `fmap` (`liftM f m = fmap f m` for any lawful monad; likewise `liftA`). bhc
/// has no per-instance `liftM`/`liftA` method, so route them through the
/// `Functor` `fmap` instance for dispatch. This lets `liftM Just p` on a user
/// monad like `ParsecT` resolve to `$instance_fmap_ParsecT` (`parsecMap`)
/// instead of falling through to the `stub: liftM` codegen path. Returns the
/// unchanged name for everything else.
/// Strip a module-alias qualifier from a builtin container's head
/// constructor: `Set.Set Extension` (under `import qualified Data.Set as
/// Set`) becomes `Set Extension`, matching the builtin Semigroup/Monoid
/// instance heads registered under the bare names. Only the head con of a
/// `Set`/`Map` spelling is rewritten; every other type is returned as-is.
fn strip_container_qualifier(ty: &Ty) -> Ty {
    fn canon_con(tc: &bhc_types::TyCon) -> Option<bhc_types::TyCon> {
        let name = tc.name.as_str();
        let last = name.rsplit('.').next()?;
        if last != name && matches!(last, "Set" | "Map") {
            Some(bhc_types::TyCon::new(Symbol::intern(last), tc.kind.clone()))
        } else {
            None
        }
    }
    match ty {
        Ty::Con(tc) => canon_con(tc).map_or_else(|| ty.clone(), Ty::Con),
        Ty::App(f, a) => match strip_container_qualifier(f) {
            new_f if &new_f != f.as_ref() => Ty::App(Box::new(new_f), a.clone()),
            _ => ty.clone(),
        },
        _ => ty.clone(),
    }
}

fn canonical_functor_method(name: Symbol) -> Symbol {
    match name.as_str() {
        // `<$>` IS `fmap` — without this it is not recognized as a class
        // method here, and `g <$> getState` in a class-default body compiled
        // to DIRECT application of `g` to the parser value (pandoc's
        // `getOption` crash).
        "liftM" | "liftA" | "<$>" => Symbol::intern("fmap"),
        _ => name,
    }
}

/// `<$>`/`fmap` and their monad/applicative spellings (`liftM`/`liftA`, already
/// canonicalized to `fmap`) take a plain function as the first operand and a
/// monadic value as the *last* operand. Only the last operand needs its
/// dictionary forced — see `lower_monad_operand`.
fn is_fmap_like_method(name: Symbol) -> bool {
    matches!(name.as_str(), "<$>" | "fmap" | "<$")
}

/// Lower an operand of a monadic bind (`>>=`/`>>`) whose monad `m` is the
/// concrete type `monad_ty` (kind `* -> *`). A constrained parser used as such
/// an operand — the `digit` in `digit >>= f` — needs its own dictionary applied
/// at the concrete stream type, exactly as an ordinary value argument does; the
/// monad-method dispatch otherwise lowers it bare, so the parser reaches
/// `parserBind` as a `\$d -> ParsecT` function and is mis-run (the `runP` crash
/// again). The operand's expected type is `monad_ty` applied to an unknown
/// result — enough to pin the stream type in the operand's own constraint
/// without needing its context-determined result type. A continuation lambda or
/// an already-concrete operand ignores the expected type and lowers normally.
fn lower_monad_operand(
    ctx: &mut LowerContext,
    arg: &hir::Expr,
    monad_ty: &Ty,
    _span: Span,
) -> LowerResult<core::Expr> {
    // The dispatched monad type can still carry variables — inside a local
    // (`let`/`where`) binding the operands are bare (`anyChar`) and pin
    // nothing, so it arrives as `ParsecT ?s ?u ?m`. Typing the operand against
    // that pins none of `s`/`u`/`m`, leaving the operand's own constraint
    // (`Stream ?s ?m Char`) unresolvable and the value emitted still awaiting
    // its dictionary. The enclosing binding's signature does determine them:
    // match against its result monad and substitute. Dispatch itself is left
    // alone — only the operand's expected type improves.
    let mut monad = monad_ty.clone();
    if has_type_variables(&monad) {
        if let Some(sig) = ctx.current_binding_sig().cloned() {
            let mut result = &sig;
            while let Ty::Fun(_, ret) = result {
                result = ret.as_ref();
            }
            if let Ty::App(sig_monad, _) = result {
                let mut subst = bhc_types::Subst::new();
                match_ty(&monad, sig_monad, &mut subst);
                monad = subst.apply(&monad);
            }
        }
    }
    let expected = Ty::App(Box::new(monad), Box::new(Ty::Error));
    lower_value_arg(ctx, arg, Some(&expected))
}

fn resolve_constrained_fn_dicts(
    ctx: &mut LowerContext,
    head_ref: &DefRef,
    args: &[&hir::Expr],
    span: Span,
) -> Option<Vec<core::Expr>> {
    let (user_constraints, scheme_ty) = {
        let scheme = ctx.lookup_scheme(head_ref.def_id)?;
        let uc: Vec<Constraint> = scheme
            .constraints
            .iter()
            .filter(|c| ctx.constraint_is_dict_passed(c))
            .cloned()
            .collect();
        if uc.is_empty() {
            return None;
        }
        (uc, scheme.ty.clone())
    };

    // Declared parameter types (the scheme type carries no dictionaries).
    let mut param_tys = Vec::new();
    let mut cur = &scheme_ty;
    while let Ty::Fun(a, r) = cur {
        param_tys.push(a.as_ref().clone());
        cur = r.as_ref();
    }

    // Instantiate the constrained type variables from the argument types.
    let mut subst = bhc_types::Subst::new();
    for (p, arg) in param_tys.iter().zip(args.iter()) {
        if let Some(at) = try_infer_arg_type(ctx, arg) {
            match_ty(p, &at, &mut subst);
        }
    }
    // Arguments alone may not pin the constrained variables — inside a
    // top-level CAF (`p1 = choice [char 'y', char 'x'] :: Parser Char`) no
    // sibling argument carries the stream type `s`, so the `Stream s m t`
    // dictionary silently failed to resolve and every argument shifted by
    // one at saturation. Typeck recorded this occurrence's instantiated
    // type; match the scheme against it to fill the remaining variables
    // (argument-driven bindings win — `match_ty` never overrides).
    if let Some(occ_ty) = ctx
        .resolved_expr_ty_opt(head_ref.span)
        .or_else(|| ctx.expr_ty_opt(head_ref.span))
    {
        if std::env::var("BHC_DBG_DICT4").is_ok() {
            eprintln!("DICT4: occ_ty at head span = {occ_ty:?}");
        }
        match_ty(&scheme_ty, &occ_ty, &mut subst);

        // The occurrence's recorded type can keep the ENCLOSING SIGNATURE'S
        // instantiation variables unresolved (`optional (char 'z')` inside
        // `poly :: Monad m => ParsecT String () m SourcePos` records
        // `ParsecT s u m (Maybe Char)` — s/u never substituted). Those same
        // variables appear in the signature's result position with their
        // concrete instantiations, so matching the occurrence's RESULT
        // against the signature's RESULT pins them (`s := [Char]`,
        // `u := ()`); `match_ty` binds pattern variables and tolerates the
        // differing final type argument. Without this the Stream dictionary
        // silently failed to resolve and every argument shifted at runtime.
        // Only when the enclosing signature is POLYMORPHIC. This fallback
        // exists to recover the SIGNATURE'S OWN instantiation variables, so a
        // monomorphic signature has nothing for it to recover — the only
        // variables it can bind are call-site-fresh ones, and it binds them to
        // whatever the enclosing binding happens to return. `runReaderT (f ())
        // 0` inside `main :: IO ()` records `f`'s occurrence as `() -> ?573
        // Int`, and matching that result against `IO ()` pinned `?573 := IO`,
        // resolving `Monad m` to the IO dictionary for a ReaderT computation.
        // `return` then compiled to IO's identity and `runReaderT` was handed a
        // bare value to call.
        if let Some(sig_ty) = ctx
            .current_binding_sig()
            .cloned()
            .filter(has_type_variables)
        {
            fn result_of(t: &Ty) -> &Ty {
                match t {
                    Ty::Fun(_, r) => result_of(r),
                    other => other,
                }
            }
            // The stored signature is already synonym-expanded (pandoc's
            // `MarkdownParser m a` = `ParsecT Sources ParserState m a`) —
            // see `lower_value_def`, which expands once per binding.
            let occ_result = result_of(&occ_ty).clone();
            let sig_result = result_of(&sig_ty).clone();
            match_ty(&occ_result, &sig_result, &mut subst);
        }
    } else if std::env::var("BHC_DBG_DICT4").is_ok() {
        eprintln!("DICT4: NO occ ty at head span {:?}", head_ref.span);
    }

    // Resolve one dictionary per constraint at the instantiated types.
    // Fixpoint apply: the signature-result fallback above binds the
    // occurrence's variables (scheme var -> occ var -> concrete is a
    // two-step chain a single-pass apply would leave at the occ var).
    let mut dicts = Vec::with_capacity(user_constraints.len());
    for c in &user_constraints {
        // `String` from a declared signature stays an unexpanded synonym and
        // never matches an instance head shaped `[tok]`; expand it.
        fn expand_string(t: &Ty) -> Ty {
            match t {
                Ty::Con(tc) if tc.name.as_str() == "String" => Ty::List(Box::new(Ty::Con(
                    bhc_types::TyCon::new(Symbol::intern("Char"), bhc_types::Kind::Star),
                ))),
                Ty::App(f, a) => Ty::App(Box::new(expand_string(f)), Box::new(expand_string(a))),
                Ty::List(inner) => Ty::List(Box::new(expand_string(inner))),
                Ty::Fun(a, b) => Ty::Fun(Box::new(expand_string(a)), Box::new(expand_string(b))),
                other => other.clone(),
            }
        }
        // Exactly two applies, not the fixpoint: the chain here is
        // structurally scheme-var -> occurrence-var -> concrete (the
        // signature fallback binds the second hop). `apply_resolved`'s
        // fixpoint re-walked the (sometimes enormous — citeproc) substituted
        // types until stable and made those modules ~50x slower to compile.
        let mut concrete_args: Vec<Ty> = c
            .args
            .iter()
            .map(|t| {
                let once = subst.apply(t);
                expand_string(&subst.apply(&once))
            })
            .collect();
        // Complete functional-dependency-determined arguments. For a
        // multi-parameter class like `Stream s t | s -> t`, matching the
        // function's parameter types only fixes `s` (the argument-carried
        // param); `t` stays a variable. Find the instance whose concrete
        // positions match and read the remaining types from it, so a
        // `Stream S t` constraint becomes the resolvable `Stream S Int`.
        if concrete_args.iter().any(has_type_variables) {
            // Compute the completed argument list as an owned value so the
            // immutable borrow of the class registry is released before the
            // mutable `resolve_dictionary` call below. Match the CONCRETE
            // positions of the constraint against the instance head to bind the
            // instance's own type variables, then apply that substitution to the
            // whole head to fill our undetermined positions. This handles both
            // concrete instances (`Stream S Int`) and parametric ones
            // (`Stream [tok] m tok`, where matching `s = [Char]` yields
            // `tok = Char`, hence the dependent `t = Char`).
            let completed: Option<Vec<Ty>> =
                ctx.class_registry()
                    .instances
                    .get(&c.class)
                    .and_then(|instances| {
                        instances.iter().find_map(|inst| {
                            if inst.instance_types.len() != concrete_args.len() {
                                return None;
                            }
                            let mut pat = Vec::new();
                            let mut tgt = Vec::new();
                            for (it, a) in inst.instance_types.iter().zip(&concrete_args) {
                                if !has_type_variables(a) {
                                    pat.push(it.clone());
                                    tgt.push(a.clone());
                                }
                            }
                            if pat.is_empty() {
                                return None;
                            }
                            let subst = bhc_types::types_match_multi(&pat, &tgt)?;
                            let filled: Vec<Ty> =
                                inst.instance_types.iter().map(|t| subst.apply(t)).collect();
                            // Merge per position: take the instance's type where it
                            // became concrete, keep ours where the instance stays
                            // parametric. `Monad m => Stream Sources m Char` fills
                            // the dependent `t = Char` while `m` — parametric in
                            // BOTH the call site and the instance — stays our
                            // variable (readWithM is polymorphic in it). The old
                            // all-or-nothing check rejected exactly that shape and
                            // the Stream dictionary was silently omitted.
                            let merged: Vec<Ty> = filled
                                .iter()
                                .zip(&concrete_args)
                                .map(|(f, ours)| {
                                    if has_type_variables(ours) && !has_type_variables(f) {
                                        f.clone()
                                    } else {
                                        ours.clone()
                                    }
                                })
                                .collect();
                            if merged.iter().zip(&concrete_args).all(|(m, ours)| m == ours) {
                                None
                            } else {
                                Some(merged)
                            }
                        })
                    });
            if let Some(completed_args) = completed {
                concrete_args = completed_args;
            }
        }
        if std::env::var("BHC_DBG_DICT4").is_ok() {
            eprintln!("DICT4: class={} args={concrete_args:?}", c.class.as_str());
        }
        let concrete = Constraint::new_multi(c.class, concrete_args, span);
        let dict = ctx.resolve_dictionary(&concrete, span);
        if std::env::var("BHC_DBG_DICT4").is_ok() {
            eprintln!("DICT4: resolved={}", dict.is_some());
            if dict.is_none() {
                if let Some(insts) = ctx.class_registry().instances.get(&c.class) {
                    for inst in insts {
                        eprintln!("DICT4:   candidate head={:?}", inst.instance_types);
                    }
                }
            }
        }
        dicts.push(dict?);
    }
    Some(dicts)
}

/// If `arg` references a constrained user *function* used as a value (passed,
/// not applied) at the known concrete type `expected_ty`, return it with its
/// dictionaries applied (a partial application, e.g. `sz $dSized_Box`). The
/// constrained type variables are recovered by matching the function's declared
/// type against `expected_ty` — the instantiation isn't visible at the bare
/// reference, only in the callee's parameter type. Returns `None` if `arg` is
/// not such a reference, is a class method, or any dictionary can't be resolved
/// at a concrete type.
fn lower_constrained_fn_value(
    ctx: &mut LowerContext,
    arg: &hir::Expr,
    expected_ty: &Ty,
) -> Option<core::Expr> {
    // Peel type applications / annotations to find the function reference.
    let mut head = arg;
    while let Expr::TypeApp(inner, _, _) | Expr::Ann(inner, _, _) = head {
        head = inner.as_ref();
    }
    let Expr::Var(def_ref) = head else {
        return None;
    };
    let name = ctx.lookup_var(def_ref.def_id)?.name;
    if ctx.is_class_method(name).is_some() {
        return None;
    }
    let (user_constraints, scheme_ty) = {
        let scheme = ctx.lookup_scheme(def_ref.def_id)?;
        let uc: Vec<Constraint> = scheme
            .constraints
            .iter()
            .filter(|c| ctx.constraint_is_dict_passed(c))
            .cloned()
            .collect();
        if uc.is_empty() {
            return None;
        }
        (uc, scheme.ty.clone())
    };
    // A scheme keeps whatever the signature was written with. A parser
    // declared `ParsecT String () m Char` never structurally matches the
    // occurrence type `ParsecT [Char] () IO Char`, and its `Stream String m
    // Char` constraint never matches the `Stream [tok] m tok` instance head —
    // so every constraint fails to resolve and the value is passed with NO
    // dictionaries at all. `runParserT` then runs the undicted lambda as a
    // parser and reads its closure header as a tag. Expand on both the type
    // and the constraint arguments.
    let scheme_ty = ctx.expand_type_aliases(&scheme_ty);
    let user_constraints: Vec<Constraint> = user_constraints
        .into_iter()
        .map(|c| {
            let args = c.args.iter().map(|a| ctx.expand_type_aliases(a)).collect();
            Constraint::new_multi(c.class, args, c.span)
        })
        .collect();

    // Recover the constrained type variables from the expected type, then
    // resolve one dictionary per user constraint at those types. Normalize the
    // `Parsec` synonym on both sides so a `ParsecT` scheme aligns with a
    // `Parsec` expected type (see `normalize_parsec`).
    let mut subst = bhc_types::Subst::new();
    match_ty(
        &normalize_parsec(&scheme_ty),
        &normalize_parsec(expected_ty),
        &mut subst,
    );

    let dicts = resolve_user_dicts(ctx, &user_constraints, &subst, def_ref.span)?;
    let var = ctx.lookup_var(def_ref.def_id).cloned()?;
    let mut result = core::Expr::Var(var, def_ref.span);
    for dict in dicts {
        result = core::Expr::App(Box::new(result), Box::new(dict), def_ref.span);
    }
    Some(result)
}

/// Resolve one dictionary per user constraint, given a `subst` that binds the
/// constrained type variables. Any functional-dependency-determined argument
/// left as a variable after substitution (e.g. `t` in `Stream s m t | s -> t`,
/// where matching only pins `s`/`m`) is completed from a matching instance
/// head: the constraint's concrete positions are matched against each instance
/// head to bind that instance's variables, and the binding is applied to the
/// whole head to fill the undetermined positions. Returns the dictionaries in
/// constraint order, or `None` if any cannot be resolved at a concrete type.
fn resolve_user_dicts(
    ctx: &mut LowerContext,
    user_constraints: &[Constraint],
    subst: &bhc_types::Subst,
    span: Span,
) -> Option<Vec<core::Expr>> {
    let mut dicts = Vec::with_capacity(user_constraints.len());
    for c in user_constraints {
        let mut concrete_args: Vec<Ty> = c.args.iter().map(|t| subst.apply(t)).collect();
        // A constraint with NOTHING pinned can still be satisfied by a
        // dictionary in scope for that very variable. pandoc's `anyChar ::
        // (Monad m, Stream s m Char, UpdateSourcePos s Char) => …` used inside
        // a `Monad m =>` binding has `Monad m` FIRST, wholly a variable;
        // instance completion below cannot help, because nothing is pinned to
        // select an instance with. Bailing there abandoned all THREE of
        // `anyChar`'s dictionaries, and the fallback then applied a PREFIX of
        // them, so every later argument landed one slot off.
        if concrete_args.iter().all(has_type_variables) {
            if let Some(dict) = ctx.dict_expr_for_class(c.class, &concrete_args, span) {
                dicts.push(dict);
                continue;
            }
        }
        if concrete_args.iter().any(has_type_variables) {
            let completed: Option<Vec<Ty>> =
                ctx.class_registry()
                    .instances
                    .get(&c.class)
                    .and_then(|instances| {
                        instances.iter().find_map(|inst| {
                            if inst.instance_types.len() != concrete_args.len() {
                                return None;
                            }
                            let mut pat = Vec::new();
                            let mut tgt = Vec::new();
                            for (it, a) in inst.instance_types.iter().zip(&concrete_args) {
                                if !has_type_variables(a) {
                                    pat.push(it.clone());
                                    tgt.push(a.clone());
                                }
                            }
                            if pat.is_empty() {
                                return None;
                            }
                            let sub = bhc_types::types_match_multi(&pat, &tgt)?;
                            let filled: Vec<Ty> =
                                inst.instance_types.iter().map(|t| sub.apply(t)).collect();
                            if !filled.iter().any(has_type_variables) {
                                return Some(filled);
                            }
                            // The instance matched every pinned argument but
                            // left one of its OWN variables standing: `instance
                            // Monad m => Stream Sources m Char` matched against
                            // `Stream Sources ? Char` still has `m`. Keep OUR
                            // argument in the open positions so the constraint
                            // names the caller's `m`, and let construction fill
                            // the instance's context from the dictionaries in
                            // scope — the same route the enclosing call already
                            // takes for `runParserT`'s own `Stream` dictionary.
                            let kept: Vec<Ty> = inst
                                .instance_types
                                .iter()
                                .zip(&concrete_args)
                                .map(|(it, ours)| {
                                    let f = sub.apply(it);
                                    if has_type_variables(&f) {
                                        ours.clone()
                                    } else {
                                        f
                                    }
                                })
                                .collect();
                            Some(kept)
                        })
                    });
            // Leave it to other paths (bare lowering) if still not concrete.
            concrete_args = completed?;
        }
        let concrete = Constraint::new_multi(c.class, concrete_args, span);
        let dict = ctx.resolve_dictionary(&concrete, span)?;
        dicts.push(dict);
    }
    Some(dicts)
}

/// If `arg` is a constrained user *function* applied to value arguments — e.g.
/// `char 'a'` or `many1 digit`, a parser combinator with its own `Stream`
/// constraint partially applied — return it with its dictionaries inserted
/// *before* the value arguments (`char $dStream 'a'`). Without this the first
/// value argument lands in the dictionary parameter's slot and every argument
/// is shifted by one, so the saturated parser reads a value where it expects a
/// dictionary (the `runP+…` crash). The constrained type variables are
/// recovered from the known result type `expected_ty` — authoritative under
/// `match_ty`'s first-binding-wins rule — and, for anything it leaves open,
/// from the value arguments' inferred types. Returns `None` when `arg` is not
/// such an application, its head is a class method, or any dictionary cannot be
/// resolved at a concrete type; the caller then falls back to plain lowering.
/// Normalize the `Parsec s u a` builtin type synonym to its expansion
/// `ParsecT s u Identity a`, so it aligns with the `ParsecT` form used by parser
/// combinators when a dictionary's type variables are recovered by matching. The
/// `parse`/`runParser` family is declared over `Parsec` (a denylisted builtin
/// synonym that is not expanded in interfaces), while the combinators use
/// `ParsecT`; without this, matching a `ParsecT s u m a` result against a
/// `Parsec s u a` expected type aligns the four-argument spine against a
/// three-argument one, so the monad `m` binds to the user state and `s`/`t` stay
/// unpinned (the `Stream` fundep then picks an arbitrary — wrong — instance).
fn normalize_parsec(ty: &Ty) -> Ty {
    let mut head = ty;
    let mut args: Vec<&Ty> = Vec::new();
    while let Ty::App(f, x) = head {
        args.push(x.as_ref());
        head = f.as_ref();
    }
    args.reverse();
    if let Ty::Con(tc) = head {
        if tc.name.as_str() == "Parsec" && args.len() == 3 {
            let star_to_star = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
            let identity = Ty::Con(TyCon::new(Symbol::intern("Identity"), star_to_star));
            let parsect_kind = Kind::Arrow(
                Box::new(Kind::Star),
                Box::new(Kind::Arrow(
                    Box::new(Kind::Star),
                    Box::new(Kind::Arrow(
                        Box::new(Kind::Star),
                        Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
                    )),
                )),
            );
            let parsect = Ty::Con(TyCon::new(Symbol::intern("ParsecT"), parsect_kind));
            // ParsecT s u Identity a
            let app = |f: Ty, x: Ty| Ty::App(Box::new(f), Box::new(x));
            return app(
                app(
                    app(app(parsect, args[0].clone()), args[1].clone()),
                    identity,
                ),
                args[2].clone(),
            );
        }
    }
    ty.clone()
}

fn lower_constrained_fn_app(
    ctx: &mut LowerContext,
    arg: &hir::Expr,
    expected_ty: &Ty,
) -> LowerResult<Option<core::Expr>> {
    // Peel the application spine: head + value arguments in source order.
    let mut head = arg;
    let mut value_args: Vec<&hir::Expr> = Vec::new();
    while let Expr::App(f, a, _) = head {
        value_args.push(a.as_ref());
        head = f.as_ref();
    }
    if value_args.is_empty() {
        // A bare reference — handled by `lower_constrained_fn_value`.
        return Ok(None);
    }
    value_args.reverse();
    // Peel type applications / annotations to reach the function reference.
    while let Expr::TypeApp(inner, _, _) | Expr::Ann(inner, _, _) = head {
        head = inner.as_ref();
    }
    let Expr::Var(def_ref) = head else {
        return Ok(None);
    };
    let Some(name) = ctx.lookup_var(def_ref.def_id).map(|v| v.name) else {
        return Ok(None);
    };
    if ctx.is_class_method(name).is_some() {
        return Ok(None);
    }
    let (user_constraints, scheme_ty) = {
        let Some(scheme) = ctx.lookup_scheme(def_ref.def_id) else {
            return Ok(None);
        };
        let uc: Vec<Constraint> = scheme
            .constraints
            .iter()
            .filter(|c| ctx.constraint_is_dict_passed(c))
            .cloned()
            .collect();
        if uc.is_empty() {
            // An unconstrained head (`manyAccum`, `option`, `count`) still needs
            // this treatment when it carries a constrained parser VALUE argument:
            // the parser must get its OWN dictionary at the concrete stream type
            // recovered from `expected_ty`, otherwise it reaches the combinator
            // as an unforced, dictionary-less value (a `\$d -> ParsecT` / arity-0
            // CAF) and is mis-run. Proceed only in that case; otherwise leave it
            // to plain lowering so ordinary applications are unaffected.
            let has_constrained_arg = value_args.iter().any(|a| references_constrained_fn(ctx, a));
            if !has_constrained_arg {
                return Ok(None);
            }
        }
        (uc, scheme.ty.clone())
    };

    // Declared parameter types (the scheme carries no dictionaries).
    let mut param_tys: Vec<Ty> = Vec::new();
    let mut cur = &scheme_ty;
    while let Ty::Fun(a, r) = cur {
        param_tys.push(a.as_ref().clone());
        cur = r.as_ref();
    }
    // Recover the constrained type variables. Match the result type after the
    // supplied value arguments against `expected_ty` first (the concrete use
    // site wins under first-binding-wins), then let the value arguments fill
    // anything the result type leaves open.
    let mut subst = bhc_types::Subst::new();
    let mut result_ty = &scheme_ty;
    for _ in 0..value_args.len() {
        if let Ty::Fun(_, r) = result_ty {
            result_ty = r.as_ref();
        }
    }
    let normalized_expected = normalize_parsec(expected_ty);
    let normalized_result = normalize_parsec(result_ty);
    match_ty(&normalized_result, &normalized_expected, &mut subst);
    for (p, a) in param_tys.iter().zip(value_args.iter()) {
        if let Some(at) = try_infer_arg_type(ctx, a) {
            match_ty(p, &at, &mut subst);
        }
    }

    let Some(dicts) = resolve_user_dicts(ctx, &user_constraints, &subst, def_ref.span) else {
        return Ok(None);
    };

    let Some(var) = ctx.lookup_var(def_ref.def_id).cloned() else {
        return Ok(None);
    };
    let mut result = core::Expr::Var(var, def_ref.span);
    for dict in dicts {
        result = core::Expr::App(Box::new(result), Box::new(dict), def_ref.span);
    }
    // Value arguments follow the dictionaries. Lower each as a value argument in
    // turn, so a nested constrained parser (the `digit` in `many1 digit`) gets
    // its own dictionary at the instantiated parameter type.
    for (i, a) in value_args.iter().enumerate() {
        let expected = param_tys.get(i).map(|p| subst.apply(p));
        let arg_core = lower_value_arg(ctx, a, expected.as_ref())?;
        result = core::Expr::App(Box::new(result), Box::new(arg_core), def_ref.span);
    }
    Ok(Some(result))
}

/// The unqualified name of a QUALIFIED monad-family operator referenced inside
/// an instance body for a non-builtin monad — `Applicative.*>` written in
/// parsec's `instance Monad (ParsecT s u m)`. `None` for everything else, so an
/// ordinary qualified name (`T.length`) is untouched.
fn instance_local_qualified_method(ctx: &LowerContext, name: Symbol) -> Option<Symbol> {
    let s = name.as_str();
    let bare = s
        .rsplit_once('.')
        .map(|(_, b)| b)
        .filter(|b| !b.is_empty())?;
    if !matches!(
        bare,
        "*>" | "<*" | "<*>" | ">>=" | ">>" | "<|>" | "<$>" | "pure" | "return" | "fmap" | "mplus"
    ) {
        return None;
    }
    let inst_ty = ctx.current_instance_type()?;
    if LowerContext::is_builtin_monad_type(inst_ty) {
        return None;
    }
    let bare = Symbol::intern(bare);
    ctx.is_class_method(bare).map(|_| bare)
}

/// Whether `expr` references a constrained user *function* used as a value
/// (a bare reference to a function with a user-class constraint, not a class
/// method and not applied here).
fn is_constrained_fn_value(ctx: &LowerContext, expr: &hir::Expr) -> bool {
    let mut h = expr;
    while let Expr::TypeApp(i, _, _) | Expr::Ann(i, _, _) = h {
        h = i.as_ref();
    }
    let Expr::Var(dr) = h else {
        return false;
    };
    if let Some(name) = ctx.lookup_var(dr.def_id).map(|v| v.name) {
        if ctx.is_class_method(name).is_some() {
            return false;
        }
    }
    ctx.lookup_scheme(dr.def_id)
        .is_some_and(|s| s.constraints.iter().any(|c| ctx.is_user_class(c.class)))
}

/// Whether `expr` is a bare constrained function value OR an *application* of one
/// whose dictionary is not yet applied — `digit`, `char '4'`, `many1 digit`.
/// Peels the whole application spine (and any type applications / annotations) to
/// the head reference. Used to decide whether an unconstrained combinator head
/// (`manyAccum`, `count`) carries a constrained parser argument that still needs
/// its own dictionary: `is_constrained_fn_value` only matches a *bare* reference,
/// so an App-form argument like `char '4'` would otherwise be lowered plainly and
/// reach the combinator as an unforced, dictionary-less arity-0 value.
fn references_constrained_fn(ctx: &LowerContext, expr: &hir::Expr) -> bool {
    let mut h = expr;
    loop {
        match h {
            Expr::App(f, _, _) => h = f.as_ref(),
            Expr::TypeApp(i, _, _) | Expr::Ann(i, _, _) => h = i.as_ref(),
            _ => break,
        }
    }
    let Expr::Var(dr) = h else {
        return false;
    };
    if let Some(name) = ctx.lookup_var(dr.def_id).map(|v| v.name) {
        if ctx.is_class_method(name).is_some() {
            return false;
        }
    }
    ctx.lookup_scheme(dr.def_id)
        .is_some_and(|s| s.constraints.iter().any(|c| ctx.is_user_class(c.class)))
}

/// Handle a call that passes a constrained function as a *value* argument, when
/// the type at which it is used is determined by the call's other arguments
/// (e.g. `twice sz (Box 3) (Box 4)` where `twice :: (a -> Int) -> a -> a -> Int`
/// — `a` is pinned by the second/third arguments, not by `twice`'s own
/// parameter for `sz`). The whole spine is needed because those sibling
/// arguments are not visible at the point the value argument alone is lowered.
///
/// Solves the callee's type variables from all inferable argument types, then
/// resolves each constrained-function-value argument's dictionaries at the
/// resulting concrete parameter type. Returns the fully lowered application, or
/// `None` (leaving the normal cases to run) when there is no such argument, the
/// callee is itself user-constrained, or any dictionary cannot be resolved.
fn try_lower_spine_with_dicts(
    ctx: &mut LowerContext,
    f: &hir::Expr,
    x: &hir::Expr,
    span: Span,
) -> LowerResult<Option<core::Expr>> {
    // Peel the full spine: callee head + arguments in source order.
    let mut head: &hir::Expr = f;
    let mut args: Vec<&hir::Expr> = vec![x];
    loop {
        match head {
            Expr::App(g, a, _) => {
                args.push(a.as_ref());
                head = g.as_ref();
            }
            Expr::TypeApp(inner, _, _) | Expr::Ann(inner, _, _) => head = inner.as_ref(),
            _ => break,
        }
    }
    args.reverse();
    let Expr::Var(callee_ref) = head else {
        return Ok(None);
    };
    // A class-method callee (`a <> b` chains) is handled by method dispatch,
    // never by this spine path. Without this, registering Semigroup/Monoid as
    // builtin classes let every `<>` application through the gate below, and
    // speculative arg lowering at each level of a long chain re-lowers the
    // nested spine combinatorially (Muse's block chains hung the compiler).
    if let Some(name) = ctx.lookup_var(callee_ref.def_id).map(|v| v.name) {
        if ctx.is_class_method(name).is_some() {
            return Ok(None);
        }
    }

    // Only act when some argument is a constrained-function value; otherwise
    // leave the existing cases untouched.
    if !args.iter().any(|a| is_constrained_fn_value(ctx, a)) {
        return Ok(None);
    }

    // Callee parameter types. A user-constrained callee is left to the
    // constrained-function call paths (Case 2/3).
    let Some(scheme) = ctx.lookup_scheme(callee_ref.def_id) else {
        return Ok(None);
    };
    if scheme
        .constraints
        .iter()
        .any(|c| ctx.is_user_class(c.class))
    {
        return Ok(None);
    }
    let mut callee_param_tys = Vec::new();
    let mut cur = &scheme.ty;
    while let Ty::Fun(p, r) = cur {
        callee_param_tys.push(p.as_ref().clone());
        cur = r.as_ref();
    }

    // Solve the callee's type variables from the inferable argument types.
    let mut subst = bhc_types::Subst::new();
    for (pty, a) in callee_param_tys.iter().zip(args.iter()) {
        if let Some(at) = try_infer_arg_type(ctx, a) {
            match_ty(pty, &at, &mut subst);
        }
    }

    let Some(callee_var) = ctx.lookup_var(callee_ref.def_id).cloned() else {
        return Ok(None);
    };
    let mut result = core::Expr::Var(callee_var, callee_ref.span);
    for (i, a) in args.iter().enumerate() {
        if is_constrained_fn_value(ctx, a) {
            let Some(expected) = callee_param_tys.get(i).map(|p| subst.apply(p)) else {
                return Ok(None);
            };
            let Some(arg_core) = lower_constrained_fn_value(ctx, a, &expected) else {
                return Ok(None);
            };
            result = core::Expr::App(Box::new(result), Box::new(arg_core), span);
        } else {
            let arg_core = lower_expr(ctx, a)?;
            result = core::Expr::App(Box::new(result), Box::new(arg_core), span);
        }
    }
    Ok(Some(result))
}

/// The type constructor at the head of an applied type: `ParsecT s u m a` has
/// head `ParsecT`.
fn applied_head_name(ty: &Ty) -> Option<Symbol> {
    match ty {
        Ty::Con(tc) => Some(tc.name),
        Ty::App(f, _) => applied_head_name(f),
        _ => None,
    }
}

/// The monad a traversal runs in, from the type typeck recorded for the whole
/// application (`m ()` for `mapM_`), else the enclosing binding's result type.
fn traversal_monad_ty(
    ctx: &LowerContext,
    span: Span,
    head_span: Span,
    n_args: usize,
) -> Option<Ty> {
    // Every source is tried in turn and the first with a CONCRETE head wins:
    // an eta-expanded point-free binding (`string = mapM char`) records an
    // application type whose head is still a variable, and stopping at it
    // declined the rewrite even though the signature named `ParsecT`.
    let mut candidates: Vec<Ty> = Vec::new();
    let from_occurrence = ctx.resolved_expr_ty_opt(span).and_then(|t| match t {
        Ty::App(m, _) => Some(*m),
        _ => None,
    });
    candidates.extend(from_occurrence.clone());
    let rest = None
        .or_else(|| {
            // Nothing recorded for the application itself — a non-final `do`
            // statement often has none. The HEAD's instantiated type carries
            // the same answer behind its arrows.
            let mut t = ctx.resolved_expr_ty_opt(head_span)?;
            for _ in 0..n_args {
                match t {
                    Ty::Fun(_, ret) => t = *ret,
                    _ => return None,
                }
            }
            match t {
                Ty::App(m, _) => Some(*m),
                _ => None,
            }
        })
        .or_else(|| {
            let mut result = ctx.current_binding_sig()?;
            while let Ty::Fun(_, ret) = result {
                result = ret.as_ref();
            }
            match result {
                Ty::App(m, _) => Some(m.as_ref().clone()),
                _ => None,
            }
        })
        // An instance method's body has neither: parsec's
        // `p1 <* p2 = do { x1 <- p1; void p2; return x1 }` is the `<*` of
        // `instance Applicative (ParsecT s u m)`, and the instance names the
        // monad. Wrong for a traversal at some OTHER monad inside an instance
        // — but then no `Monad` instance resolves at this type and the rewrite
        // declines anyway.
        .or_else(|| ctx.current_instance_type().cloned());
    candidates.extend(rest);
    candidates
        .iter()
        .find(|t| applied_head_name(t).is_some())
        .or_else(|| candidates.first())
        .cloned()
}

/// A monad whose actions codegen runs as it builds them. For those, and only
/// those, `mapM_ f xs` really is `map f xs` with the result dropped.
fn monad_runs_eagerly(ty: &Ty) -> bool {
    matches!(
        applied_head_name(ty)
            .map(|n| n.as_str().to_string())
            .as_deref(),
        Some("IO" | "StateT" | "ReaderT" | "ExceptT" | "WriterT" | "Identity" | "Maybe" | "Either")
    )
}

fn nullary_var(name: &str, span: Span) -> core::Expr {
    core::Expr::Var(
        Var {
            name: Symbol::intern(name),
            id: VarId::new(0),
            ty: Ty::Error,
        },
        span,
    )
}

/// `mapM_`, `forM_`, `mapM`, `forM`, `traverse`, `traverse_`, `sequence` and
/// `sequence_` at a monad that is NOT run eagerly.
///
/// codegen lowers `mapM_ f xs` to `map f xs` and returns null. That is right
/// for bhc's IO, where building an action IS running it — and wrong everywhere
/// else: at `ParsecT` each `f x` is a parser VALUE, so mapping builds a list of
/// parsers that nobody runs, and the null travels on as the action. pandoc's
/// `checkNotes` and `reportLogMessages` end in `mapM_`, so `readMarkdown`
/// segfaulted on an EMPTY document.
///
/// Rewritten to a fold through the monad's own `>>=` and `pure`. `foldr` is
/// fine to build the chain in any order: constructing an action of a
/// non-eager monad has no effect.
/// What `lower_monadic_lift` does with the bound results.
#[derive(Clone, Copy)]
enum MonadicShape {
    /// `pure ()` — `void`.
    Unit,
    /// `pure (f v1 .. vn)` — `liftM`, `liftM2`.
    ApplyFn,
}

/// `void a`, `liftM f a`, `liftM2 f a b` at a monad codegen does not run
/// eagerly: bind each action in order, then `pure` the result.
fn lower_monadic_lift(
    ctx: &mut LowerContext,
    args: &[&hir::Expr],
    n_actions: usize,
    shape: MonadicShape,
    head_span: Span,
    span: Span,
) -> LowerResult<Option<core::Expr>> {
    let Some(monad_ty) = traversal_monad_ty(ctx, span, head_span, args.len()) else {
        return Ok(None);
    };
    if monad_runs_eagerly(&monad_ty) || applied_head_name(&monad_ty).is_none() {
        return Ok(None);
    }
    let Some(bind_e) = ctx.resolve_method_at_concrete_type(
        Symbol::intern(">>="),
        Symbol::intern("Monad"),
        &monad_ty,
        span,
    ) else {
        return Ok(None);
    };
    let Some(pure_e) = ctx.resolve_method_at_concrete_type(
        Symbol::intern("pure"),
        Symbol::intern("Applicative"),
        &monad_ty,
        span,
    ) else {
        return Ok(None);
    };
    // The actions are the LAST `n_actions` arguments; anything before them is
    // the function `liftM`/`liftM2` applies.
    let split = args.len() - n_actions;
    let fn_core = match shape {
        MonadicShape::ApplyFn => Some(lower_expr(ctx, args[0])?),
        MonadicShape::Unit => None,
    };
    let mut action_cores = Vec::with_capacity(n_actions);
    for a in &args[split..] {
        action_cores.push(lower_expr(ctx, a)?);
    }
    let binders: Vec<Var> = (0..n_actions)
        .map(|i| ctx.fresh_var(&format!("$lift_v{i}"), Ty::Error, span))
        .collect();
    let result = match shape {
        MonadicShape::Unit => nullary_var("()", span),
        MonadicShape::ApplyFn => {
            let mut e = fn_core.expect("ApplyFn always lowers its function");
            for b in &binders {
                e = core::Expr::App(
                    Box::new(e),
                    Box::new(core::Expr::Var(b.clone(), span)),
                    span,
                );
            }
            e
        }
    };
    let mut body = core::Expr::App(Box::new(pure_e), Box::new(result), span);
    for (action, binder) in action_cores.into_iter().zip(binders).rev() {
        body = core::Expr::App(
            Box::new(core::Expr::App(
                Box::new(bind_e.clone()),
                Box::new(action),
                span,
            )),
            Box::new(core::Expr::Lam(binder, Box::new(body), span)),
            span,
        );
    }
    Ok(Some(body))
}

/// `\mf mx -> mf >>= \g -> mx >>= \v -> pure (g v)` at `monad_ty`: `ap` used
/// as a value, which is how parsec defines `<*>` for `ParsecT`.
fn lower_ap_lambda(ctx: &mut LowerContext, monad_ty: &Ty, span: Span) -> Option<core::Expr> {
    let bind_e = ctx.resolve_method_at_concrete_type(
        Symbol::intern(">>="),
        Symbol::intern("Monad"),
        monad_ty,
        span,
    )?;
    let pure_e = ctx.resolve_method_at_concrete_type(
        Symbol::intern("pure"),
        Symbol::intern("Applicative"),
        monad_ty,
        span,
    )?;
    let mf = ctx.fresh_var("$ap_mf", Ty::Error, span);
    let mx = ctx.fresh_var("$ap_mx", Ty::Error, span);
    let g = ctx.fresh_var("$ap_g", Ty::Error, span);
    let v = ctx.fresh_var("$ap_v", Ty::Error, span);
    let app2 = |h: core::Expr, a: core::Expr, b: core::Expr| {
        core::Expr::App(
            Box::new(core::Expr::App(Box::new(h), Box::new(a), span)),
            Box::new(b),
            span,
        )
    };
    let applied = core::Expr::App(
        Box::new(core::Expr::Var(g.clone(), span)),
        Box::new(core::Expr::Var(v.clone(), span)),
        span,
    );
    let inner = app2(
        bind_e.clone(),
        core::Expr::Var(mx.clone(), span),
        core::Expr::Lam(
            v,
            Box::new(core::Expr::App(Box::new(pure_e), Box::new(applied), span)),
            span,
        ),
    );
    let body = app2(
        bind_e,
        core::Expr::Var(mf.clone(), span),
        core::Expr::Lam(g, Box::new(inner), span),
    );
    Some(core::Expr::Lam(
        mf,
        Box::new(core::Expr::Lam(mx, Box::new(body), span)),
        span,
    ))
}

/// `ap mf mx` = `mf >>= \g -> mx >>= \v -> pure (g v)`, at a monad codegen
/// does not run eagerly.
fn lower_monadic_ap(
    ctx: &mut LowerContext,
    mf: &hir::Expr,
    mx: &hir::Expr,
    head_span: Span,
    span: Span,
) -> LowerResult<Option<core::Expr>> {
    let Some(monad_ty) = traversal_monad_ty(ctx, span, head_span, 2) else {
        return Ok(None);
    };
    if monad_runs_eagerly(&monad_ty) || applied_head_name(&monad_ty).is_none() {
        return Ok(None);
    }
    let Some(bind_e) = ctx.resolve_method_at_concrete_type(
        Symbol::intern(">>="),
        Symbol::intern("Monad"),
        &monad_ty,
        span,
    ) else {
        return Ok(None);
    };
    let Some(pure_e) = ctx.resolve_method_at_concrete_type(
        Symbol::intern("pure"),
        Symbol::intern("Applicative"),
        &monad_ty,
        span,
    ) else {
        return Ok(None);
    };
    let mf_core = lower_expr(ctx, mf)?;
    let mx_core = lower_expr(ctx, mx)?;
    let g = ctx.fresh_var("$ap_g", Ty::Error, span);
    let v = ctx.fresh_var("$ap_v", Ty::Error, span);
    let app2 = |h: core::Expr, a: core::Expr, b: core::Expr| {
        core::Expr::App(
            Box::new(core::Expr::App(Box::new(h), Box::new(a), span)),
            Box::new(b),
            span,
        )
    };
    let applied = core::Expr::App(
        Box::new(core::Expr::Var(g.clone(), span)),
        Box::new(core::Expr::Var(v.clone(), span)),
        span,
    );
    let inner = app2(
        bind_e.clone(),
        mx_core,
        core::Expr::Lam(
            v,
            Box::new(core::Expr::App(Box::new(pure_e), Box::new(applied), span)),
            span,
        ),
    );
    Ok(Some(app2(
        bind_e,
        mf_core,
        core::Expr::Lam(g, Box::new(inner), span),
    )))
}

fn try_lower_monadic_traversal(
    ctx: &mut LowerContext,
    f: &hir::Expr,
    x: &hir::Expr,
    span: Span,
) -> LowerResult<Option<core::Expr>> {
    // Spine: either `Var name a1 a2` (two arguments) or `Var name a1` (one).
    let (head_ref, args): (&DefRef, Vec<&hir::Expr>) = match f {
        hir::Expr::Var(dr) => (dr, vec![x]),
        hir::Expr::App(inner, a1, _) => match inner.as_ref() {
            hir::Expr::Var(dr) => (dr, vec![a1.as_ref(), x]),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    let Some(name) = ctx.lookup_var(head_ref.def_id).map(|v| v.name) else {
        return Ok(None);
    };
    // `ap mf mx` is `<*>` written through the Monad; codegen has no
    // implementation for it and aborts with `stub: ap not implemented`.
    if name.as_str() == "ap" && args.len() == 2 {
        return lower_monadic_ap(ctx, args[0], args[1], head_ref.span, span);
    }
    // `void`, `liftM` and `liftM2` are the same shape: bind each action, then
    // `pure` of something built from the results. parsec's `<*` is
    // `do { x <- p; void q; return x }`, so `anyChar <* anyChar` crashed.
    // `when c a` / `unless c a` — codegen's versions run the action as they
    // build it, so at `ParsecT` the parser ran during construction and the
    // caller got whatever it returned.
    if let ("when" | "unless", 2) = (name.as_str(), args.len()) {
        if let Some(monad_ty) = traversal_monad_ty(ctx, span, head_ref.span, 2) {
            if !monad_runs_eagerly(&monad_ty) && applied_head_name(&monad_ty).is_some() {
                if let Some(pure_e) = ctx.resolve_method_at_concrete_type(
                    Symbol::intern("pure"),
                    Symbol::intern("Applicative"),
                    &monad_ty,
                    span,
                ) {
                    let cond = lower_expr(ctx, args[0])?;
                    let action = lower_expr(ctx, args[1])?;
                    let skip =
                        core::Expr::App(Box::new(pure_e), Box::new(nullary_var("()", span)), span);
                    let (then_e, else_e) = if name.as_str() == "when" {
                        (action, skip)
                    } else {
                        (skip, action)
                    };
                    return Ok(Some(make_if_expr(cond, then_e, else_e, span)));
                }
            }
        }
    }
    if let Some((n_actions, build)) = match (name.as_str(), args.len()) {
        ("void", 1) => Some((1usize, MonadicShape::Unit)),
        ("liftM" | "fmapM", 2) => Some((1, MonadicShape::ApplyFn)),
        ("liftM2", 3) => Some((2, MonadicShape::ApplyFn)),
        _ => None,
    } {
        return lower_monadic_lift(ctx, &args, n_actions, build, head_ref.span, span);
    }
    // `collect`: the result list is kept. `elems`: the arguments are already
    // actions, there is no function to apply.
    // `missing_list`: the function is there but the container is not, which is
    // how a point-free definition arrives — `string = mapM char` is eta
    // expanded only after lowering, so HIR holds `mapM char` alone. The
    // rewrite supplies the container as a lambda parameter.
    let (collect, elems, flipped, missing_list) = match (name.as_str(), args.len()) {
        ("mapM_" | "traverse_" | "mapA_", 2) => (false, false, false, false),
        ("mapM_" | "traverse_" | "mapA_", 1) => (false, false, false, true),
        ("mapM" | "traverse", 2) => (true, false, false, false),
        ("mapM" | "traverse", 1) => (true, false, false, true),
        ("forM_" | "for_", 2) => (false, false, true, false),
        ("forM" | "for", 2) => (true, false, true, false),
        ("sequence_" | "sequenceA_", 1) => (false, true, false, false),
        ("sequence" | "sequenceA", 1) => (true, true, false, false),
        _ => return Ok(None),
    };
    let n_arrows = if missing_list {
        args.len() + 1
    } else {
        args.len()
    };
    let monad_ty = match traversal_monad_ty(ctx, span, head_ref.span, n_arrows) {
        Some(t) => t,
        None => {
            if std::env::var("BHC_DBG_TRAV").is_ok() {
                eprintln!("trav {}: no monad type", name.as_str());
            }
            return Ok(None);
        }
    };
    if std::env::var("BHC_DBG_TRAV").is_ok() {
        eprintln!(
            "trav {}: monad {:?}",
            name.as_str(),
            applied_head_name(&monad_ty)
        );
    }
    if monad_runs_eagerly(&monad_ty) || applied_head_name(&monad_ty).is_none() {
        return Ok(None);
    }
    let Some(bind_e) = ctx.resolve_method_at_concrete_type(
        Symbol::intern(">>="),
        Symbol::intern("Monad"),
        &monad_ty,
        span,
    ) else {
        return Ok(None);
    };
    let Some(pure_e) = ctx.resolve_method_at_concrete_type(
        Symbol::intern("pure"),
        Symbol::intern("Applicative"),
        &monad_ty,
        span,
    ) else {
        return Ok(None);
    };

    let (fn_expr, list_expr) = if missing_list {
        (Some(args[0]), None)
    } else if elems {
        (None, Some(args[0]))
    } else if flipped {
        (Some(args[1]), Some(args[0]))
    } else {
        (Some(args[0]), Some(args[1]))
    };
    let list_param = if missing_list {
        Some(ctx.fresh_var("$trav_xs", Ty::Error, span))
    } else {
        None
    };
    let list_core = match (list_expr, &list_param) {
        (Some(e), _) => lower_expr(ctx, e)?,
        (None, Some(v)) => core::Expr::Var(v.clone(), span),
        (None, None) => return Ok(None),
    };
    let fn_core = match fn_expr {
        Some(e) => Some(lower_expr(ctx, e)?),
        None => None,
    };

    let app2 = |g: core::Expr, a: core::Expr, b: core::Expr| {
        core::Expr::App(
            Box::new(core::Expr::App(Box::new(g), Box::new(a), span)),
            Box::new(b),
            span,
        )
    };

    let elem = ctx.fresh_var("$trav_x", Ty::Error, span);
    let rest = ctx.fresh_var("$trav_k", Ty::Error, span);
    // `f x`, or the element itself when it already is the action.
    let action = match fn_core {
        Some(g) => core::Expr::App(
            Box::new(g),
            Box::new(core::Expr::Var(elem.clone(), span)),
            span,
        ),
        None => core::Expr::Var(elem.clone(), span),
    };

    let step_body = if collect {
        let v = ctx.fresh_var("$trav_v", Ty::Error, span);
        let vs = ctx.fresh_var("$trav_vs", Ty::Error, span);
        let cons = app2(
            nullary_var(":", span),
            core::Expr::Var(v.clone(), span),
            core::Expr::Var(vs.clone(), span),
        );
        let inner = app2(
            bind_e.clone(),
            core::Expr::Var(rest.clone(), span),
            core::Expr::Lam(
                vs,
                Box::new(core::Expr::App(
                    Box::new(pure_e.clone()),
                    Box::new(cons),
                    span,
                )),
                span,
            ),
        );
        app2(
            bind_e.clone(),
            action,
            core::Expr::Lam(v, Box::new(inner), span),
        )
    } else {
        let ignored = ctx.fresh_var("$trav_ignored", Ty::Error, span);
        app2(
            bind_e.clone(),
            action,
            core::Expr::Lam(ignored, Box::new(core::Expr::Var(rest.clone(), span)), span),
        )
    };

    let step = core::Expr::Lam(
        elem,
        Box::new(core::Expr::Lam(rest, Box::new(step_body), span)),
        span,
    );
    let seed = core::Expr::App(
        Box::new(pure_e),
        Box::new(if collect {
            nullary_var("[]", span)
        } else {
            nullary_var("()", span)
        }),
        span,
    );
    let folded = core::Expr::App(
        Box::new(app2(nullary_var("foldr", span), step, seed)),
        Box::new(list_core),
        span,
    );
    Ok(Some(match list_param {
        Some(v) => core::Expr::Lam(v, Box::new(folded), span),
        None => folded,
    }))
}

/// Lower a function application, handling dictionary-passing for class methods
/// and constrained functions when the argument type is known.
///
/// When `f` is a class method or constrained function and we can infer the
/// argument type, we resolve dictionaries at this concrete type. This handles
/// cases like `describe Red` where `describe` is a class method of `Describable`
/// and `Red` is a `Color` constructor.
fn lower_app(
    ctx: &mut LowerContext,
    f: &hir::Expr,
    x: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    // `mapM_`/`mapM`/`sequence_`/… at a monad codegen does not run eagerly.
    if let Some(result) = try_lower_monadic_traversal(ctx, f, x, span)? {
        return Ok(result);
    }

    // A constrained function passed as a value argument: resolve its
    // dictionaries from the type at which the call uses it (determined by this
    // call's arguments). Handles both a concrete callee parameter and a
    // polymorphic one pinned by sibling arguments. Only fires when such an
    // argument is present; otherwise the normal cases below run unchanged.
    if let Some(result) = try_lower_spine_with_dicts(ctx, f, x, span)? {
        return Ok(result);
    }

    // Check if f is a Var referencing a class method or constrained function
    if let Expr::Var(def_ref) = f {
        if let Some(var) = ctx.lookup_var(def_ref.def_id).cloned() {
            let method_name = canonical_functor_method(var.name);

            // Case 1: Class method resolution
            if let Some(class_name) = ctx.is_class_method(method_name) {
                let is_user = ctx.is_user_class(class_name);
                let is_monad_family = ctx.is_monad_family_class(class_name);

                // Case 1a: Dictionary in scope (from existential pattern match).
                // Select the method from the dictionary and apply the argument.
                let dict_matches = in_scope_dict_matches(ctx, class_name, span);
                if is_user && dict_matches {
                    if let Some(dict_var) = ctx.lookup_dict(class_name).cloned() {
                        if let Some(method_expr) =
                            ctx.select_method_from_dict(&dict_var, class_name, method_name, span)
                        {
                            let x_core = lower_expr(ctx, x)?;
                            return Ok(core::Expr::App(
                                Box::new(method_expr),
                                Box::new(x_core),
                                span,
                            ));
                        }
                    }
                }

                // The occurrence's own monad decides when the dictionary in
                // scope is for a DIFFERENT one. pandoc's `getsCommonState` —
                // a PandocMonad method with a class default — is called
                // inside a parser from a `PandocMonad m =>` binding, and
                // selecting it from the dictionary for `m` handed the parser
                // an action belonging to the base monad.
                if is_user && !dict_matches && ctx.lookup_dict(class_name).is_some() {
                    if let Some(method_expr) = ctx.select_method_by_result_type(
                        class_name,
                        method_name,
                        def_ref.def_id,
                        def_ref.span,
                    ) {
                        let x_core = lower_expr(ctx, x)?;
                        return Ok(core::Expr::App(
                            Box::new(method_expr),
                            Box::new(x_core),
                            span,
                        ));
                    }
                }

                // Case 1a': the monad family supplied as a SUPERCLASS, single
                // argument. `return x` arrives here while `a >>= k` arrives at
                // the multi-argument site below; patching only one leaves the
                // other bare for codegen to guess an ambient layer for.
                if is_monad_family
                    && crate::dictionary::monad_witness_enabled()
                    && ctx.lookup_dict(class_name).is_none()
                {
                    if let Some(method_expr) =
                        ctx.select_method_via_superclass(class_name, method_name, span)
                    {
                        let x_core = lower_expr(ctx, x)?;
                        return Ok(core::Expr::App(
                            Box::new(method_expr),
                            Box::new(x_core),
                            span,
                        ));
                    }
                }

                // Case 1b: No dict in scope — resolve via instance lookup
                if (is_user || is_monad_family)
                    && (ctx.lookup_dict(class_name).is_none() || !dict_matches)
                {
                    let param_count = ctx.class_param_count(class_name);

                    if param_count > 1 && is_user {
                        // Multi-param class with just one argument (user classes only).
                        // Try to infer the arg type and complete remaining
                        // types from matching instances (fundep-style).
                        if let Some(arg_ty) = try_infer_arg_type(ctx, x) {
                            let mut types_for_resolution = vec![arg_ty];
                            // Search instances to complete the type list
                            if let Some(instances) = ctx.class_registry().instances.get(&class_name)
                            {
                                for inst in instances {
                                    if inst.instance_types.len() >= param_count {
                                        let all_match = types_for_resolution
                                            .iter()
                                            .enumerate()
                                            .all(|(i, ty)| inst.instance_types.get(i) == Some(ty));
                                        if all_match {
                                            types_for_resolution =
                                                inst.instance_types[..param_count].to_vec();
                                            break;
                                        }
                                    }
                                }
                            }
                            if types_for_resolution.len() >= param_count {
                                if let Some(method_expr) = ctx.resolve_method_at_concrete_types(
                                    method_name,
                                    class_name,
                                    &types_for_resolution,
                                    span,
                                ) {
                                    let x_core = lower_expr(ctx, x)?;
                                    return Ok(core::Expr::App(
                                        Box::new(method_expr),
                                        Box::new(x_core),
                                        span,
                                    ));
                                }
                            }
                        }
                        // Fall through to Case 3 if we couldn't resolve
                    } else {
                        // Single-param class: resolve at concrete type from argument.
                        //
                        // `pure`/`return` are the exception: their class parameter is
                        // the RESULT constructor (`pure :: a -> f a`), never the
                        // argument `a`. Inferring the instance from the argument type
                        // resolves the wrong instance — and when no instance exists at
                        // the argument type, dictionary construction can still fabricate
                        // a bogus dictionary, yielding an identity-like miscompile (e.g.
                        // `item >>= \a -> pure a` lowering the lambda to `\a -> a` with a
                        // dangling closure capture). Leave `inferred` as `None` for them
                        // so control falls through to the result-type dispatch in Case
                        // 1.5 below, which is where `return` already resolves correctly.
                        let is_result_determined =
                            matches!(method_name.as_str(), "pure" | "return");
                        let inferred = if is_result_determined {
                            None
                        } else {
                            try_infer_arg_type(ctx, x).or_else(|| {
                                // Fallback: use monad context stack for >>=/>>/return/pure
                                if is_monad_family {
                                    ctx.current_monad_type().cloned().or_else(|| {
                                        // Last resort: recover the monad constructor from
                                        // this method application's own fixpoint-resolved
                                        // type `N b` (strip the value arg). Lets a
                                        // user/derived monad's `>>=` dispatch when the
                                        // operands' types are themselves unresolved — e.g.
                                        // `return 5 >>= \x -> ...` in a top-level do-block
                                        // over a GND newtype, where both operands are
                                        // as-yet-undispatched `return`s.
                                        match ctx.resolved_expr_ty_opt(span) {
                                            Some(Ty::App(head, _)) => Some(*head),
                                            _ => None,
                                        }
                                    })
                                } else {
                                    None
                                }
                            })
                        };
                        if let Some(concrete_ty) = inferred {
                            // For monad-family builtin classes, skip if the concrete type
                            // is a builtin monad — let codegen handle the fast path
                            if is_monad_family
                                && !is_user
                                && LowerContext::is_builtin_monad_type(&concrete_ty)
                            {
                                // Fall through to codegen fast path
                            } else {
                                let resolved = ctx.resolve_method_at_concrete_type(
                                    method_name,
                                    class_name,
                                    &concrete_ty,
                                    span,
                                );
                                if let Some(method_expr) = resolved {
                                    let x_core = lower_expr(ctx, x)?;
                                    return Ok(core::Expr::App(
                                        Box::new(method_expr),
                                        Box::new(x_core),
                                        span,
                                    ));
                                }
                                // Fallback: bare type didn't match instance head.
                                // Try applied type for parameterized instances.
                                if let Some(applied_ty) = try_infer_applied_type(ctx, x) {
                                    if !(is_monad_family
                                        && !is_user
                                        && LowerContext::is_builtin_monad_type(&applied_ty))
                                    {
                                        if let Some(method_expr) = ctx
                                            .resolve_method_at_concrete_type(
                                                method_name,
                                                class_name,
                                                &applied_ty,
                                                span,
                                            )
                                        {
                                            let x_core = lower_expr(ctx, x)?;
                                            return Ok(core::Expr::App(
                                                Box::new(method_expr),
                                                Box::new(x_core),
                                                span,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Case 1.5: return/pure with monad type context
            // `return` is a standalone builtin (not a class method), so it bypasses
            // dictionary dispatch. When inside a do-block for a user-defined monad,
            // resolve it via the Applicative dictionary using the monad context stack.
            {
                let is_return_or_pure =
                    method_name.as_str() == "return" || method_name.as_str() == "pure";
                if is_return_or_pure {
                    // Candidate monad types, in priority order:
                    //  1. the enclosing do-block monad, if `>>=`/`>>` pushed one;
                    //  2. the monad constructor read off this `return`/`pure`
                    //     application's OWN resolved type `M a` (strip the value
                    //     arg `a`). (2) is what lets a bare `return x :: M a` with
                    //     no surrounding do-block dispatch — e.g. the argument in
                    //     `runIOorExplode (return 42)`, whose single-pass expr
                    //     type leaves the monad a variable (`App(Var, Int)`) but
                    //     whose fixpoint-resolved type is `App(PandocIO, Int)`.
                    let mut candidates: Vec<Ty> = Vec::new();
                    if let Some(m) = ctx.current_monad_type().cloned() {
                        candidates.push(m);
                    }
                    if let Some(Ty::App(head, _arg)) = ctx.resolved_expr_ty_opt(span) {
                        candidates.push(*head);
                    }
                    let pure_sym = Symbol::intern("pure");
                    let applicative_sym = Symbol::intern("Applicative");
                    //  3. the enclosing binding's signature. Inside a constrained
                    //     binding neither of the above pins the monad — the
                    //     do-block's own type stays a variable and the surrounding
                    //     `>>=` may itself be lowered before its monad context is
                    //     pushed — so `return` stayed a builtin returning its
                    //     argument raw. The result then travelled on as though it
                    //     were a parser. Gated like the bind's own fallback: only a
                    //     CONSUMER of the instance may dispatch, never the module
                    //     implementing it.
                    if let Some(sig) = ctx.current_binding_sig() {
                        let mut result = sig;
                        while let Ty::Fun(_, ret) = result {
                            result = ret.as_ref();
                        }
                        if let Ty::App(monad, _) = result {
                            let mut head = monad.as_ref();
                            while let Ty::App(f2, _) = head {
                                head = f2.as_ref();
                            }
                            if let Ty::Con(con) = head {
                                if ctx.has_imported_instance(applicative_sym, con.name) {
                                    candidates.push(monad.as_ref().clone());
                                }
                            }
                        }
                    }
                    //  4. the instance being defined. parsec's `<*` is
                    //     `do { x1 <- p1; void p2; return x1 }` inside
                    //     `instance Applicative (ParsecT s u m)`; none of the
                    //     above pins the monad there, so `return x1` stayed a
                    //     builtin and handed the Char on raw — a later
                    //     dereference of `0x61`.
                    if let Some(inst_ty) = ctx.current_instance_type().cloned() {
                        candidates.push(inst_ty);
                    }
                    for monad_ty in candidates {
                        // Builtin monads (IO/StateT/…) take codegen's fast path.
                        if LowerContext::is_builtin_monad_type(&monad_ty) {
                            continue;
                        }
                        if let Some(method_expr) = ctx.resolve_method_at_concrete_type(
                            pure_sym,
                            applicative_sym,
                            &monad_ty,
                            span,
                        ) {
                            let x_core = lower_expr(ctx, x)?;
                            return Ok(core::Expr::App(
                                Box::new(method_expr),
                                Box::new(x_core),
                                span,
                            ));
                        }
                    }

                    //  4. The monad is the enclosing binding's own constrained
                    //     type VARIABLE — `readMarkdown :: PandocMonad m => …
                    //     -> m Pandoc` whose body is `return doc`. Every
                    //     candidate above is concrete, so all of them miss, and
                    //     the builtin fast path has nothing to dispatch ON:
                    //     `return`'s monad appears only in its RESULT, so no
                    //     argument carries it at runtime. The builtin then
                    //     returns its argument raw and the caller unwraps a
                    //     plain value as though it were an action — `evalStateT
                    //     f 0` forces the literal 7 as a pointer.
                    //
                    //     There is a dictionary in scope for exactly this
                    //     variable: the binding's own. Select `pure` out of it,
                    //     hopping superclasses (PandocMonad ⊃ Monad ⊃
                    //     Applicative) to reach it. Last, so a concrete monad is
                    //     always preferred and previously-dispatching call sites
                    //     are untouched.
                    let monad_is_type_var = match ctx.resolved_expr_ty_opt(span) {
                        Some(Ty::App(head, _)) => matches!(head.as_ref(), Ty::Var(_)),
                        _ => false,
                    };
                    if monad_is_type_var {
                        if let Some(method_expr) =
                            ctx.select_method_via_superclass(applicative_sym, pure_sym, span)
                        {
                            let x_core = lower_expr(ctx, x)?;
                            return Ok(core::Expr::App(
                                Box::new(method_expr),
                                Box::new(x_core),
                                span,
                            ));
                        }
                    }
                }
            }

            // Case 2: A constrained user function applied to one argument.
            // Instantiate the constrained type variables by matching the
            // function's parameter types against the inferred argument type,
            // resolve a dictionary per constraint, and pass the dictionaries
            // before the argument (the definition is `\$d -> \x -> ...`).
            // `resolve_constrained_fn_dicts` returns None — falling through to
            // later cases — when the head is not user-constrained or any
            // dictionary cannot be resolved (rather than committing to a bare,
            // dictionary-less application as the old code did).
            //
            // Class methods are excluded: their own class constraint must be
            // turned into a dictionary *selection* (`$sel_N $dict`) by the
            // method-selection path, not passed as an ordinary argument.
            if ctx.is_class_method(method_name).is_none() {
                if let Some(dicts) = resolve_constrained_fn_dicts(ctx, def_ref, &[x], span) {
                    let mut result = core::Expr::Var(var.clone(), def_ref.span);
                    for dict in dicts {
                        result = core::Expr::App(Box::new(result), Box::new(dict), span);
                    }
                    let x_core = lower_expr(ctx, x)?;
                    return Ok(core::Expr::App(Box::new(result), Box::new(x_core), span));
                }
            }
        }
    }

    // Case 3: f is an App chain whose head is a class method or constrained function
    // e.g. myMap (+1) (Box 42) → f = App(Var(myMap), (+1)), x = App(Con(Box), Lit(42))
    // We peel the chain to find the head Var, then resolve dictionaries from x's type.
    if let Some((head_def_ref, collected_args)) = peel_app_chain(f) {
        if let Some(var) = ctx.lookup_var(head_def_ref.def_id).cloned() {
            let method_name = canonical_functor_method(var.name);
            if std::env::var("BHC_DBG_FMAP").is_ok() && is_fmap_like_method(method_name) {
                eprintln!(
                    "[fmap] head={} is_method={:?} args={} occ={:?}",
                    method_name,
                    ctx.is_class_method(method_name),
                    collected_args.len(),
                    ctx.resolved_expr_ty_opt(x.span())
                        .or_else(|| ctx.expr_ty_opt(x.span()))
                );
            }

            // Case 3a: Head is a class method (user-defined or monad-family)
            if let Some(class_name) = ctx.is_class_method(method_name) {
                let is_user = ctx.is_user_class(class_name);
                let is_monad_family = ctx.is_monad_family_class(class_name);

                // Dict in scope (from existential pattern): select method from dict,
                // then apply all collected args + final arg
                if is_user {
                    if let Some(dict_var) = ctx.lookup_dict(class_name).cloned() {
                        if let Some(method_expr) =
                            ctx.select_method_from_dict(&dict_var, class_name, method_name, span)
                        {
                            let mut result = method_expr;
                            for arg in &collected_args {
                                let arg_core = lower_expr(ctx, arg)?;
                                result =
                                    core::Expr::App(Box::new(result), Box::new(arg_core), span);
                            }
                            let x_core = lower_expr(ctx, x)?;
                            return Ok(core::Expr::App(Box::new(result), Box::new(x_core), span));
                        }
                    }
                }

                // The monad family supplied as a SUPERCLASS. parsec's
                // `runPT :: Stream s m t => …` has no `Monad m` constraint of
                // its own — `class Monad m => Stream s m t` supplies it — so
                // `lookup_dict(Monad)` misses and the do-block's `>>=` was
                // emitted bare for codegen to guess an ambient layer for.
                // `select_method_via_superclass` now checks that the hop is
                // about the occurrence's own monad before taking it, so a
                // ParsecT-level binding under the same constraint is left
                // alone. `a >>= k` arrives here, at the MULTI-argument site.
                if is_monad_family
                    && crate::dictionary::monad_witness_enabled()
                    && ctx.lookup_dict(class_name).is_none()
                {
                    if let Some(method_expr) =
                        ctx.select_method_via_superclass(class_name, method_name, span)
                    {
                        let mut result = method_expr;
                        for arg in &collected_args {
                            let arg_core = lower_expr(ctx, arg)?;
                            result = core::Expr::App(Box::new(result), Box::new(arg_core), span);
                        }
                        let x_core = lower_expr(ctx, x)?;
                        return Ok(core::Expr::App(Box::new(result), Box::new(x_core), span));
                    }
                }

                if (is_user || is_monad_family)
                    && (ctx.lookup_dict(class_name).is_none()
                        || !in_scope_dict_matches(ctx, class_name, span))
                {
                    let param_count = ctx.class_param_count(class_name);

                    if param_count > 1 && is_user {
                        // Multi-param class: collect types from all arguments (user classes only)
                        // For `combine Red Circle`, collected_args=[Red], x=Circle
                        // We need types from each arg: [Color, Shape]
                        let mut all_args: Vec<&hir::Expr> = collected_args.clone();
                        all_args.push(x);

                        let mut concrete_types: Vec<Ty> = Vec::new();
                        for arg in &all_args {
                            if let Some(ty) = try_infer_arg_type(ctx, arg) {
                                concrete_types.push(ty);
                            }
                        }

                        // If we have fewer types than params, try completing
                        // from instance declarations (fundep-style resolution).
                        // E.g., for `class Extract a b | a -> b` with `extract :: a -> b`,
                        // calling `extract w` gives us only type `a` from the value arg.
                        // We search instances to find the matching `b`.
                        let mut types_for_resolution = concrete_types.clone();
                        if types_for_resolution.len() < param_count
                            && !types_for_resolution.is_empty()
                        {
                            if let Some(instances) = ctx.class_registry().instances.get(&class_name)
                            {
                                for inst in instances {
                                    if inst.instance_types.len() >= param_count {
                                        let all_match = types_for_resolution
                                            .iter()
                                            .enumerate()
                                            .all(|(i, ty)| inst.instance_types.get(i) == Some(ty));
                                        if all_match {
                                            types_for_resolution =
                                                inst.instance_types[..param_count].to_vec();
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        if types_for_resolution.len() >= param_count {
                            if let Some(method_expr) = ctx.resolve_method_at_concrete_types(
                                method_name,
                                class_name,
                                &types_for_resolution,
                                span,
                            ) {
                                // Apply all arguments after dictionary resolution
                                let mut result = method_expr;
                                for arg in &collected_args {
                                    let arg_core = lower_expr(ctx, arg)?;
                                    result =
                                        core::Expr::App(Box::new(result), Box::new(arg_core), span);
                                }
                                let x_core = lower_expr(ctx, x)?;
                                return Ok(core::Expr::App(
                                    Box::new(result),
                                    Box::new(x_core),
                                    span,
                                ));
                            }
                        }
                    } else {
                        // Single-param class: resolve at the class parameter type.
                        // For multi-arg methods like `runEval :: e -> String -> String`,
                        // the first collected arg carries the instance type, not the
                        // final argument. Prefer collected args over the final arg.
                        let inferred_args = collected_args
                            .iter()
                            .find_map(|arg| try_infer_arg_type(ctx, arg));
                        let inferred_x = try_infer_arg_type(ctx, x);
                        // Semigroup/Monoid parameterize over the VALUE type
                        // itself (Inlines), not a type constructor — the
                        // span-recorded result type IS the class parameter.
                        let is_value_class = matches!(class_name.as_str(), "Semigroup" | "Monoid");
                        let inferred = inferred_args.or(inferred_x).or_else(|| {
                            // Fallback: use monad context stack for nested >>=/>>/return
                            if is_value_class {
                                ctx.resolved_expr_ty_opt(span).filter(has_concrete_head)
                            } else if is_monad_family {
                                ctx.current_monad_type()
                                    .cloned()
                                    .or_else(|| {
                                        if std::env::var("BHC_NO_CART").is_ok() {
                                            return None;
                                        }
                                        // A FULLY-CONCRETE applied operand type
                                        // (`manyTill p $eta` with sig-typed
                                        // params) carries the complete monad
                                        // instantiation — dispatch AND operand
                                        // dicting both work from it. Bare
                                        // operands (`digit` in `digit <|>
                                        // letter`) pin nothing and stay on the
                                        // codegen fast paths.
                                        collected_args
                                            .iter()
                                            .copied()
                                            .chain(std::iter::once(x))
                                            .find_map(|arg| {
                                                let full = concrete_applied_result_ty(ctx, arg)?;
                                                (!LowerContext::is_builtin_monad_type(&full))
                                                    .then_some(full)
                                            })
                                    })
                                    .or_else(|| {
                                        if std::env::var("BHC_NO_PRONGB").is_ok() {
                                            return None;
                                        }
                                        // Inside a CONSTRAINED function whose own
                                        // dict params satisfy an operand's
                                        // constraints (`many1TillChar p = fmap
                                        // T.pack . many1Till p` — `many1Till`
                                        // needs the same `Stream` dict the
                                        // enclosing fn received), dispatch by the
                                        // operand's scheme-result HEAD; operand
                                        // dicting then resolves from those
                                        // in-scope dicts. Unconstrained contexts
                                        // (Main-level `digit <|> letter`) have no
                                        // dicts in scope and keep the codegen
                                        // fast paths; a VAR result head (runPT's
                                        // bind at generic m) also skips.
                                        collected_args
                                            .iter()
                                            .copied()
                                            .chain(std::iter::once(x))
                                            .find_map(|arg| {
                                                let mut head = arg;
                                                let mut nargs = 0usize;
                                                while let hir::Expr::App(f2, _, _) = head {
                                                    nargs += 1;
                                                    head = f2.as_ref();
                                                }
                                                if nargs == 0 {
                                                    return None;
                                                }
                                                let hir::Expr::Var(dr) = head else {
                                                    return None;
                                                };
                                                let scheme = ctx.lookup_scheme(dr.def_id)?;
                                                let dict_scoped = scheme
                                                    .constraints
                                                    .iter()
                                                    .any(|c| ctx.lookup_dict(c.class).is_some());
                                                if !dict_scoped {
                                                    return None;
                                                }
                                                let mut t = scheme.ty.clone();
                                                for _ in 0..nargs {
                                                    let Ty::Fun(_, r) = t else {
                                                        return None;
                                                    };
                                                    t = *r;
                                                }
                                                let mut h = &t;
                                                while let Ty::App(f2, _) = h {
                                                    h = f2.as_ref();
                                                }
                                                (matches!(h, Ty::Con(_))
                                                    && !LowerContext::is_builtin_monad_type(h))
                                                .then(|| h.clone())
                                            })
                                    })
                                    .or_else(|| {
                                        // Last resort: recover the monad constructor from
                                        // this method application's own fixpoint-resolved
                                        // type `N b` (strip the value arg). Lets a
                                        // user/derived monad's `>>=` dispatch when the
                                        // operands' types are themselves unresolved — e.g.
                                        // `return 5 >>= \x -> ...` in a top-level do-block
                                        // over a GND newtype, where both operands are
                                        // as-yet-undispatched `return`s.
                                        match ctx.resolved_expr_ty_opt(span) {
                                            Some(Ty::App(head, _)) => Some(*head),
                                            _ => None,
                                        }
                                    })
                                    .or_else(|| {
                                        // Nothing above pinned the monad: inside a
                                        // constrained binding the operands can be bare
                                        // parameters (`manyTill p end = scan where scan
                                        // = do { x <- p; … }`), which carry no
                                        // instantiated type of their own. Left
                                        // undispatched, `>>=` stays a builtin whose
                                        // generic bind ignores parser failure.
                                        //
                                        // The binding's signature names the monad
                                        // applied to its (still variable) arguments,
                                        // which is what the parametric instance
                                        // matches. Only a CONSUMER of the instance may
                                        // use it: inside the module implementing the
                                        // instance, this would rewrite the generic
                                        // implementation — parsec's own `parserBind` —
                                        // into a call to itself, miscompiling the
                                        // library. An imported instance means we are a
                                        // consumer. A variable result head, as in
                                        // `runPT`'s bind at a generic `m`, has no head
                                        // constructor and skips.
                                        let sig = ctx.current_binding_sig()?;
                                        let mut result = sig;
                                        while let Ty::Fun(_, ret) = result {
                                            result = ret.as_ref();
                                        }
                                        let Ty::App(monad, _) = result else {
                                            return None;
                                        };
                                        let mut head = monad.as_ref();
                                        while let Ty::App(f2, _) = head {
                                            head = f2.as_ref();
                                        }
                                        let Ty::Con(con) = head else {
                                            return None;
                                        };
                                        (!LowerContext::is_builtin_monad_type(head)
                                            && ctx.has_imported_instance(class_name, con.name))
                                        .then(|| monad.as_ref().clone())
                                    })
                            } else {
                                None
                            }
                        });

                        // A dispatch type recovered from an operand's scheme is the
                        // bare head constructor (`ParsecT`), which names the monad but
                        // leaves it unapplied — and instance resolution matches on the
                        // applied form, so it found nothing and the method stayed an
                        // undispatched builtin `>>=`. At runtime that generic bind ran
                        // its continuation even after the parser had failed, so inside
                        // a constrained binding `char '!'` appeared to succeed on 'a'.
                        // The enclosing signature supplies the applied form —
                        // `ParsecT s u m`, arguments still variables, which is exactly
                        // what the parametric instance matches.
                        let inferred = inferred.map(|ty| {
                            let Ty::Con(ref con) = ty else {
                                return ty;
                            };
                            let Some(sig) = ctx.current_binding_sig() else {
                                return ty;
                            };
                            let mut result = sig;
                            while let Ty::Fun(_, ret) = result {
                                result = ret.as_ref();
                            }
                            let Ty::App(monad, _) = result else {
                                return ty;
                            };
                            let mut head = monad.as_ref();
                            while let Ty::App(f2, _) = head {
                                head = f2.as_ref();
                            }
                            match head {
                                Ty::Con(h) if h.name == con.name => monad.as_ref().clone(),
                                _ => ty,
                            }
                        });
                        if let Some(concrete_ty) = inferred {
                            // `mconcat :: [a] -> a` — the class parameter is the
                            // list's ELEMENT type, but the argument-driven
                            // inference sees the list. Unwrap it.
                            let concrete_ty = if method_name.as_str() == "mconcat" {
                                match &concrete_ty {
                                    Ty::List(t) => t.as_ref().clone(),
                                    t => t.clone(),
                                }
                            } else {
                                concrete_ty
                            };
                            // For monad-family builtin classes, skip if the concrete type
                            // is a builtin monad — let codegen handle the fast path
                            let is_builtin_m = LowerContext::is_builtin_monad_type(&concrete_ty);
                            if is_monad_family && !is_user && is_builtin_m {
                                // Fall through to codegen fast path
                            } else {
                                let mut resolved = ctx.resolve_method_at_concrete_type(
                                    method_name,
                                    class_name,
                                    &concrete_ty,
                                    span,
                                );
                                // `x <$ p` with no dedicated instance method:
                                // rewrite through the instance's `fmap` as
                                // `fmap (const x) p` (the class default).
                                let mut const_rewrite = false;
                                if resolved.is_none() && method_name.as_str() == "<$" {
                                    resolved = ctx.resolve_method_at_concrete_type(
                                        Symbol::intern("fmap"),
                                        class_name,
                                        &concrete_ty,
                                        span,
                                    );
                                    const_rewrite = resolved.is_some();
                                }
                                // `mconcat xs` with no dedicated instance
                                // method: its class default is
                                // `foldr mappend mempty xs`.
                                if resolved.is_none() && method_name.as_str() == "mconcat" {
                                    let mappend_e = ctx.resolve_method_at_concrete_type(
                                        Symbol::intern("mappend"),
                                        class_name,
                                        &concrete_ty,
                                        span,
                                    );
                                    let mempty_e = ctx.resolve_method_at_concrete_type(
                                        Symbol::intern("mempty"),
                                        class_name,
                                        &concrete_ty,
                                        span,
                                    );
                                    if let (Some(mappend_e), Some(mempty_e)) = (mappend_e, mempty_e)
                                    {
                                        let foldr_var = Var {
                                            name: Symbol::intern("foldr"),
                                            id: VarId::new(0),
                                            ty: Ty::Error,
                                        };
                                        resolved = Some(core::Expr::App(
                                            Box::new(core::Expr::App(
                                                Box::new(core::Expr::Var(foldr_var, span)),
                                                Box::new(mappend_e),
                                                span,
                                            )),
                                            Box::new(mempty_e),
                                            span,
                                        ));
                                    }
                                }
                                if let Some(method_expr) = resolved {
                                    // Push monad context for >>=/>>) so return/pure
                                    // inside lambda bodies resolve via dictionary dispatch
                                    let is_bind_op = method_name.as_str() == ">>="
                                        || method_name.as_str() == ">>";
                                    // Operators whose operands are monadic VALUES
                                    // (`m a <|> m a`, `m a *> m b`, `digit >>= f`)
                                    // — a constrained parser operand must get its
                                    // own dictionary applied, else it reaches the
                                    // instance method (parserPlus/parserBind) as an
                                    // unforced `\$d -> ParsecT` / thunk and is
                                    // mis-run.
                                    let operands_are_monadic = is_bind_op
                                        || matches!(
                                            method_name.as_str(),
                                            "<|>" | "mplus" | "*>" | "<*" | "<*>"
                                        );
                                    // `<$>`/`fmap` (and `liftM`/`liftA`, already
                                    // canonicalized to `fmap`) take a plain function
                                    // first and the monadic value LAST. Only the last
                                    // operand needs its dictionary forced — the
                                    // function operand stays plain. Without this,
                                    // `fmap f digit` reaches `parsecMap` with `digit`
                                    // an unforced `\$d -> ParsecT` CAF and segfaults,
                                    // exactly as `<|>` did.
                                    let is_fmap_like = is_fmap_like_method(method_name);
                                    if is_bind_op {
                                        ctx.push_monad_type(concrete_ty.clone());
                                    }
                                    let mut result = method_expr;
                                    for arg in &collected_args {
                                        let arg_core = if operands_are_monadic {
                                            lower_monad_operand(ctx, arg, &concrete_ty, span)?
                                        } else {
                                            lower_expr(ctx, arg)?
                                        };
                                        let arg_core = if const_rewrite {
                                            // `<$`'s value operand becomes fmap's
                                            // function operand: \_ -> x
                                            let ignored = ctx.fresh_var("_const", Ty::Error, span);
                                            core::Expr::Lam(ignored, Box::new(arg_core), span)
                                        } else {
                                            arg_core
                                        };
                                        result = core::Expr::App(
                                            Box::new(result),
                                            Box::new(arg_core),
                                            span,
                                        );
                                    }
                                    let x_core = if operands_are_monadic || is_fmap_like {
                                        lower_monad_operand(ctx, x, &concrete_ty, span)?
                                    } else {
                                        lower_expr(ctx, x)?
                                    };
                                    if is_bind_op {
                                        ctx.pop_monad_type();
                                    }
                                    return Ok(core::Expr::App(
                                        Box::new(result),
                                        Box::new(x_core),
                                        span,
                                    ));
                                }
                                // Fallback: try applied type for parameterized instances
                                let applied = try_infer_applied_type(ctx, x).or_else(|| {
                                    collected_args
                                        .iter()
                                        .find_map(|arg| try_infer_applied_type(ctx, arg))
                                });
                                if let Some(applied_ty) = applied {
                                    if !(is_monad_family
                                        && !is_user
                                        && LowerContext::is_builtin_monad_type(&applied_ty))
                                    {
                                        if let Some(method_expr) = ctx
                                            .resolve_method_at_concrete_type(
                                                method_name,
                                                class_name,
                                                &applied_ty,
                                                span,
                                            )
                                        {
                                            // Push monad context for applied type too
                                            let is_bind_op = method_name.as_str() == ">>="
                                                || method_name.as_str() == ">>";
                                            if is_bind_op {
                                                ctx.push_monad_type(applied_ty.clone());
                                            }
                                            let mut result = method_expr;
                                            for arg in &collected_args {
                                                let arg_core = lower_expr(ctx, arg)?;
                                                result = core::Expr::App(
                                                    Box::new(result),
                                                    Box::new(arg_core),
                                                    span,
                                                );
                                            }
                                            let x_core = lower_expr(ctx, x)?;
                                            if is_bind_op {
                                                ctx.pop_monad_type();
                                            }
                                            return Ok(core::Expr::App(
                                                Box::new(result),
                                                Box::new(x_core),
                                                span,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Case 3a': a monad-family method at a still-POLYMORPHIC monad,
            // with a user-class dictionary IN SCOPE whose class lists the
            // method's class as a SUPERCLASS — pandoc's readMarkdown do-block
            // binds at generic `m` with only `$dPandocMonad` available, and
            // `Monad` is a direct superclass of `PandocMonad`. Select the
            // method from the dictionary's superclass slot; operands lower
            // plainly (no concrete type to dict them at). Without this the
            // bind fails to lower and the enclosing case collapses into an
            // unconditional non-exhaustive throw. Guard on the occurrence
            // type staying unresolved so concrete binds (IO inside the same
            // constrained function) keep their fast paths.
            // Inside the operands of a superclass-dispatched bind (flag set
            // below), `return`/`pure` must come from the SAME dictionary —
            // otherwise they lower as identity and the continuation yields a
            // raw value where an action is expected. Scoped to that dynamic
            // extent only: a global mapping mis-selected from parsec's
            // `Stream` dicts (whose superclasses also carry Monad) and broke
            // every combinator probe.
            if ctx.superclass_bind_depth() > 0 && matches!(method_name.as_str(), "return" | "pure")
            {
                if let Some(method_expr) = ctx.select_method_via_superclass(
                    Symbol::intern("Applicative"),
                    Symbol::intern("pure"),
                    span,
                ) {
                    let mut result = method_expr;
                    for arg in &collected_args {
                        let arg_core = lower_expr(ctx, arg)?;
                        result = core::Expr::App(Box::new(result), Box::new(arg_core), span);
                    }
                    let x_core = lower_expr(ctx, x)?;
                    return Ok(core::Expr::App(Box::new(result), Box::new(x_core), span));
                }
            }
            if let Some(class_name) = ctx.is_class_method(method_name) {
                if ctx.is_monad_family_class(class_name)
                    && (ctx.lookup_dict(class_name).is_none()
                        || !in_scope_dict_matches(ctx, class_name, span))
                {
                    let occ_unresolved = ctx.resolved_expr_ty_opt(span).is_none_or(|t| {
                        let mut h = &t;
                        while let Ty::App(f, _) = h {
                            h = f;
                        }
                        !matches!(h, Ty::Con(_))
                    });
                    if occ_unresolved {
                        if let Some(method_expr) =
                            ctx.select_method_via_superclass(class_name, method_name, span)
                        {
                            // While lowering this bind's operands, nested
                            // `return`/`pure` at the same generic monad must
                            // select from the same dictionary (see the
                            // return/pure hook above).
                            ctx.enter_superclass_bind();
                            let mut lowered = Vec::with_capacity(collected_args.len() + 1);
                            for arg in &collected_args {
                                lowered.push(lower_expr(ctx, arg));
                            }
                            let x_res = lower_expr(ctx, x);
                            ctx.exit_superclass_bind();
                            let mut result = method_expr;
                            for arg_core in lowered {
                                result =
                                    core::Expr::App(Box::new(result), Box::new(arg_core?), span);
                            }
                            return Ok(core::Expr::App(Box::new(result), Box::new(x_res?), span));
                        }
                    }
                }
            }

            // Case 3b: Head is a constrained function with unresolved type-variable constraints
            if let Some(scheme) = ctx.lookup_scheme(head_def_ref.def_id) {
                let has_unresolved = scheme
                    .constraints
                    .iter()
                    .any(|c| ctx.is_user_class(c.class) && c.args.iter().any(has_type_variables));
                if has_unresolved {
                    // Clone the constraint list before any mutable ctx call so the
                    // immutable `scheme` borrow is released (its last use is here).
                    let constraints = scheme.constraints.clone();
                    // Preferred path: resolve one dictionary per user constraint
                    // by matching the callee's parameter types against ALL
                    // argument types (and completing functional-dependency
                    // params). This handles multi-parameter classes such as
                    // `Stream s t | s -> t`, which the single-type fallback
                    // below builds as a malformed one-arg constraint that never
                    // resolves — leaving the dictionary silently omitted and
                    // every subsequent argument shifted by one.
                    let mut all_args: Vec<&hir::Expr> = collected_args.clone();
                    all_args.push(x);
                    // Only for constrained *functions*, not class methods. A
                    // method whose instance cannot be resolved at a concrete
                    // type here — e.g. a superclass method applied to polymorphic
                    // arguments inside another constrained function — must fall
                    // through to method dispatch, not have its dictionary passed
                    // as an ordinary argument (`myEqual dict x y`), a form
                    // codegen does not lower and stubs at runtime.
                    if ctx.is_class_method(var.name).is_none() {
                        if let Some(dicts) =
                            resolve_constrained_fn_dicts(ctx, head_def_ref, &all_args, span)
                        {
                            // Instantiated parameter types, so a constrained
                            // *value* argument (a parser with its own `Stream`
                            // constraint, passed to `parse`) gets its dictionary
                            // applied at the concrete type — not just the callee.
                            let param_info =
                                callee_param_tys_and_subst(ctx, head_def_ref, &all_args);
                            let mut result = core::Expr::Var(var.clone(), head_def_ref.span);
                            for dict in dicts {
                                result = core::Expr::App(Box::new(result), Box::new(dict), span);
                            }
                            for (i, arg) in all_args.iter().enumerate() {
                                let expected = param_info
                                    .as_ref()
                                    .and_then(|(ptys, sub)| ptys.get(i).map(|p| sub.apply(p)));
                                let arg_core = lower_value_arg(ctx, arg, expected.as_ref())?;
                                result =
                                    core::Expr::App(Box::new(result), Box::new(arg_core), span);
                            }
                            return Ok(result);
                        }
                    }

                    // Try to infer concrete type from x or collected args
                    let inferred = try_infer_arg_type(ctx, x).or_else(|| {
                        collected_args
                            .iter()
                            .find_map(|arg| try_infer_arg_type(ctx, arg))
                    });
                    if let Some(concrete_ty) = inferred {
                        // Build from the head var (not lowered f, to avoid double resolution)
                        let mut result = core::Expr::Var(var.clone(), head_def_ref.span);

                        // Instantiate the callee's DECLARED constraints through its
                        // own parameter types. The guess below — "whichever argument
                        // first yields a type" — is only right when the class
                        // parameter happens to be that argument: `setMeta :: (HasMeta
                        // a, ToMetaValue b) => Text -> b -> a -> a` guessed `[Char]`
                        // from the Text, and `withQuoteContext :: HasQuoteContext st m
                        // => QuoteContext -> …` guessed `QuoteContext`, neither of
                        // which is the class parameter. It also flattened every
                        // multi-parameter class to one argument, which no instance
                        // head can match.
                        let declared_subst =
                            callee_param_tys_and_subst(ctx, head_def_ref, &all_args)
                                .map(|(_, sub)| sub);

                        for constraint in &constraints {
                            if ctx.is_user_class(constraint.class)
                                && constraint.args.iter().any(has_type_variables)
                            {
                                // Try the substituted form first, then the guess. This
                                // is strictly additive: a call site the guess already
                                // resolved keeps resolving, so no argument list that
                                // used to be well-formed can start shifting. (The
                                // guess is still wrong wherever the class parameter is
                                // genuinely polymorphic — `addMeta field val = modify
                                // (setMeta field val)` resolves `ToMetaValue` at the
                                // caller's `Text` — but replacing a wrong dictionary
                                // with a missing one trades a wrong answer for a
                                // crash, which is not an improvement.)
                                let instantiated = declared_subst.as_ref().and_then(|sub| {
                                    let args: Vec<Ty> =
                                        constraint.args.iter().map(|a| sub.apply(a)).collect();
                                    args.iter().any(|a| !has_type_variables(a)).then(|| {
                                        Constraint::new_multi(
                                            constraint.class,
                                            args,
                                            constraint.span,
                                        )
                                    })
                                });
                                let guessed = Constraint::new(
                                    constraint.class,
                                    concrete_ty.clone(),
                                    constraint.span,
                                );
                                let resolved = instantiated
                                    .and_then(|c| ctx.resolve_dictionary(&c, span))
                                    .or_else(|| ctx.resolve_dictionary(&guessed, span));
                                if let Some(dict_expr) = resolved {
                                    result = core::Expr::App(
                                        Box::new(result),
                                        Box::new(dict_expr),
                                        span,
                                    );
                                } else {
                                    // The callee's lambda has a dict parameter for
                                    // this constraint; omitting the argument shifts
                                    // every later argument one slot at runtime.
                                    // Name the callee and its byte range: the
                                    // bare class name gives no way to find which
                                    // of a module's call sites is at fault, and
                                    // these warnings are the main signal for
                                    // argument-shift miscompiles.
                                    eprintln!(
                                        "warning: dictionary for `{}` could not be resolved at call site of `{}` ({}..{}); argument slots may shift",
                                        constraint.class.as_str(),
                                        var.name.as_str(),
                                        span.lo.as_u32(),
                                        span.hi.as_u32()
                                    );
                                }
                            }
                        }

                        // Apply collected args, then x
                        for arg in &collected_args {
                            let arg_core = lower_expr(ctx, arg)?;
                            result = core::Expr::App(Box::new(result), Box::new(arg_core), span);
                        }
                        let x_core = lower_expr(ctx, x)?;
                        return Ok(core::Expr::App(Box::new(result), Box::new(x_core), span));
                    }
                }
            }
        }
    }

    // For existential constructors, insert dictionary arguments.
    // When a constructor like `MkDesc Foo` has existential constraints
    // (e.g., `Describable d`), we need: `MkDesc dict_Describable_Foo Foo`
    if let Expr::Con(def_ref) = f {
        if let Some(info) = ctx.lookup_constructor(def_ref.def_id).cloned() {
            if info.existential_dict_count > 0 {
                let mut result = lower_expr(ctx, f)?;
                // Infer the argument type to construct existential constraints
                if let Some(arg_ty) = try_infer_arg_type(ctx, x) {
                    for class_name in &info.existential_classes {
                        let constraint = Constraint::new(*class_name, arg_ty.clone(), span);
                        if let Some(dict_expr) = ctx.resolve_dictionary(&constraint, def_ref.span) {
                            result = core::Expr::App(Box::new(result), Box::new(dict_expr), span);
                        }
                    }
                }
                let x_core = lower_expr(ctx, x)?;
                return Ok(core::Expr::App(Box::new(result), Box::new(x_core), span));
            }
        }
    }

    // Default: lower f and x normally
    let f_core = lower_expr(ctx, f)?;
    let x_core = lower_expr(ctx, x)?;
    Ok(core::Expr::App(Box::new(f_core), Box::new(x_core), span))
}

/// Lower a type application expression.
///
/// Type applications like `f @Int` are used to instantiate polymorphic functions
/// at specific types. For class methods, this is the key mechanism for resolving
/// which instance to use at a monomorphic call site.
///
/// For example, `(+) @Int` should resolve to the `(+)` method from the `Num Int` instance.
fn lower_type_app(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    ty: &Ty,
    span: Span,
) -> LowerResult<core::Expr> {
    // Check if this is a type application to a class method
    if let Expr::Var(def_ref) = expr {
        if let Some(var) = ctx.lookup_var(def_ref.def_id) {
            let method_name = canonical_functor_method(var.name);

            // Check if this is a class method that needs dictionary resolution.
            // Applies to user-defined classes and monad-family classes at non-builtin types.
            if let Some(class_name) = ctx.is_class_method(method_name) {
                let is_user = ctx.is_user_class(class_name);
                let is_monad_family = ctx.is_monad_family_class(class_name);
                if is_user || (is_monad_family && !LowerContext::is_builtin_monad_type(ty)) {
                    if let Some(method_expr) =
                        ctx.resolve_method_at_concrete_type(method_name, class_name, ty, span)
                    {
                        return Ok(method_expr);
                    }
                }
                // Fall through to regular handling if resolution fails
            }
        }
    }

    // Regular type application handling
    let expr_core = lower_expr(ctx, expr)?;
    Ok(core::Expr::TyApp(Box::new(expr_core), ty.clone(), span))
}

/// Lower a constructor reference to Core.
fn lower_con(ctx: &mut LowerContext, def_ref: &DefRef) -> LowerResult<core::Expr> {
    // Constructors are represented as variables in Core
    // (they get special treatment during optimization)
    if let Some(var) = ctx.lookup_var(def_ref.def_id) {
        Ok(core::Expr::Var(var.clone(), def_ref.span))
    } else {
        let placeholder = Var {
            name: Symbol::intern("Con"),
            id: VarId::new(def_ref.def_id.index()),
            ty: Ty::Error,
        };
        Ok(core::Expr::Var(placeholder, def_ref.span))
    }
}

/// Lower a lambda expression to Core.
///
/// HIR lambdas can have multiple patterns: `\x y -> body`
/// Core lambdas take a single variable, so we need to:
/// 1. Create nested lambdas for each argument
/// 2. Compile patterns into case expressions
fn lower_lambda(
    ctx: &mut LowerContext,
    pats: &[hir::Pat],
    body: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    if pats.is_empty() {
        // No patterns - just lower the body
        return lower_expr(ctx, body);
    }

    // First pass: register all pattern variables so they're available in the body
    // We need to do this before lowering the body because the body may reference them
    let mut pat_vars: Vec<(hir::DefId, Var)> = Vec::new();
    for pat in pats {
        register_pattern_vars(ctx, pat, &mut pat_vars);
    }

    // Now lower the body (pattern vars are registered)
    let body_core = lower_expr(ctx, body)?;

    // Build nested lambdas from right to left
    let mut result = body_core;

    for pat in pats.iter().rev() {
        // Check if the pattern is simple (just a variable)
        match pat {
            hir::Pat::Var(name, def_id, _) => {
                // Simple case: pattern is just a variable
                // Look up the var we registered earlier
                let var = ctx.lookup_var(*def_id).cloned().unwrap_or_else(|| Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                });
                result = core::Expr::Lam(var, Box::new(result), span);
            }
            hir::Pat::Wild(_) => {
                // Wildcard: just use a fresh variable that's not referenced
                let arg_var = ctx.fresh_var("lam", Ty::Error, span);
                result = core::Expr::Lam(arg_var, Box::new(result), span);
            }
            _ => {
                // Complex pattern: need a case expression
                let arg_var = ctx.fresh_var("lam", Ty::Error, span);
                let alt = lower_pat_to_alt(ctx, pat, result.clone(), span)?;
                let default_alt = Alt {
                    con: AltCon::Default,
                    binders: vec![],
                    rhs: make_pattern_error(span),
                };

                let case_expr = core::Expr::Case(
                    Box::new(core::Expr::Var(arg_var.clone(), span)),
                    vec![alt, default_alt],
                    Ty::Error,
                    span,
                );

                result = core::Expr::Lam(arg_var, Box::new(case_expr), span);
            }
        }
    }

    Ok(result)
}

/// Register all variables bound by a pattern into the context.
fn register_pattern_vars(
    ctx: &mut LowerContext,
    pat: &hir::Pat,
    vars: &mut Vec<(hir::DefId, Var)>,
) {
    match pat {
        hir::Pat::Var(name, def_id, _) => {
            let var = Var {
                name: *name,
                id: VarId::new(def_id.index()),
                ty: Ty::Error,
            };
            ctx.register_var(*def_id, var.clone());
            vars.push((*def_id, var));
        }
        hir::Pat::As(name, def_id, inner, _) => {
            let var = Var {
                name: *name,
                id: VarId::new(def_id.index()),
                ty: Ty::Error,
            };
            ctx.register_var(*def_id, var.clone());
            vars.push((*def_id, var));
            register_pattern_vars(ctx, inner, vars);
        }
        hir::Pat::Con(_, sub_pats, _) => {
            for sub in sub_pats {
                register_pattern_vars(ctx, sub, vars);
            }
        }
        hir::Pat::RecordCon(_, field_pats, _) => {
            for fp in field_pats {
                register_pattern_vars(ctx, &fp.pat, vars);
            }
        }
        hir::Pat::Or(left, right, _) => {
            register_pattern_vars(ctx, left, vars);
            register_pattern_vars(ctx, right, vars);
        }
        hir::Pat::Ann(inner, _, _) | hir::Pat::View(_, inner, _) => {
            register_pattern_vars(ctx, inner, vars);
        }
        hir::Pat::Wild(_) | hir::Pat::Lit(_, _) | hir::Pat::Error(_) => {}
    }
}

/// Lower a let expression to Core.
fn lower_let(
    ctx: &mut LowerContext,
    bindings: &[hir::Binding],
    body: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    use crate::binding::preregister_bindings;

    // First, pre-register all binding variables so they're available
    // when lowering the body (and for recursive references in RHSes)
    let _vars = preregister_bindings(ctx, bindings)?;

    // Now lower the body - it can reference the bound variables
    let body_core = lower_expr(ctx, body)?;

    // Check if we have pattern bindings that need case expressions
    // For simple `let x = e in body`, we just create a let binding.
    // For pattern bindings like `let (x, y) = e in body`, we generate
    // `case e of (x, y) -> body` instead.
    lower_let_bindings(ctx, bindings, body_core, span)
}

/// Lower let bindings, handling pattern bindings with case expressions.
/// Collect `(name, def_id, span)` for every binder in a pattern.
fn pattern_binders(pat: &hir::Pat, out: &mut Vec<(Symbol, DefId, Span)>) {
    match pat {
        hir::Pat::Wild(_) | hir::Pat::Lit(_, _) | hir::Pat::Error(_) => {}
        hir::Pat::Var(name, def_id, span) => out.push((*name, *def_id, *span)),
        hir::Pat::Con(_, pats, _) => {
            for p in pats {
                pattern_binders(p, out);
            }
        }
        hir::Pat::RecordCon(_, field_pats, _) => {
            for fp in field_pats {
                pattern_binders(&fp.pat, out);
            }
        }
        hir::Pat::As(name, def_id, inner, span) => {
            out.push((*name, *def_id, *span));
            pattern_binders(inner, out);
        }
        hir::Pat::Or(l, r, _) => {
            pattern_binders(l, out);
            pattern_binders(r, out);
        }
        hir::Pat::Ann(inner, _, _) | hir::Pat::View(_, inner, _) => pattern_binders(inner, out),
    }
}

/// Desugar a mixed let/where group: every non-Var pattern binding becomes a
/// fresh whole-value Var binding plus one Var binding per pattern binder that
/// cases on the whole value (see `lower_let_bindings`).
fn desugar_pattern_bindings(
    ctx: &mut LowerContext,
    bindings: &[hir::Binding],
) -> Vec<hir::Binding> {
    let mut out = Vec::with_capacity(bindings.len() * 2);
    for b in bindings {
        if matches!(b.pat, hir::Pat::Var(..)) {
            out.push(b.clone());
            continue;
        }
        let pb = ctx.fresh_var("$pb", Ty::Error, b.span);
        // Synthetic DefId range distinct from HIR ids and the 800k/900k
        // interface-synthesis ranges.
        let pb_def = DefId::new(1_600_000 + pb.id.index());
        ctx.register_var(pb_def, pb.clone());
        out.push(hir::Binding {
            pat: hir::Pat::Var(pb.name, pb_def, b.span),
            sig: b.sig.clone(),
            rhs: b.rhs.clone(),
            span: b.span,
        });
        let mut binders = Vec::new();
        pattern_binders(&b.pat, &mut binders);
        for (name, def_id, bspan) in binders {
            let selector = hir::Expr::Case(
                Box::new(hir::Expr::Var(hir::DefRef {
                    def_id: pb_def,
                    span: b.span,
                })),
                vec![hir::CaseAlt {
                    pat: b.pat.clone(),
                    guards: Vec::new(),
                    rhs: hir::Expr::Var(hir::DefRef {
                        def_id,
                        span: bspan,
                    }),
                    span: b.span,
                }],
                b.span,
            );
            out.push(hir::Binding {
                pat: hir::Pat::Var(name, def_id, bspan),
                sig: None,
                rhs: selector,
                span: b.span,
            });
        }
    }
    out
}

fn lower_let_bindings(
    ctx: &mut LowerContext,
    bindings: &[hir::Binding],
    body: core::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    use crate::binding::collect_free_vars;

    // Haskell `let`/`where` groups are mutually recursive. When the group is
    // entirely simple variable bindings, a binding may reference a SIBLING
    // defined later in the group — e.g. parsec's `tokens` has
    // `walk [] rs = ok rs` with `ok` bound *after* `walk`. The naive
    // per-binding nesting below binds each on its own, placing an earlier
    // binding OUTSIDE a later one it references, so the forward reference is
    // out of scope and codegen emits `stub: ok not implemented`.
    //
    // The fix is dependency analysis: split the group into strongly-connected
    // components and emit each SCC in topological order, with a component's
    // dependencies bound OUTER (so they are in scope for its RHSes). A
    // singleton SCC with no self-edge is a `NonRec`; a self-recursive singleton
    // or a genuine mutual cycle is a `Bind::Rec`. This keeps mutually-recursive
    // groups to true cycles only (codegen mislowers a multi-binding `Rec` whose
    // members capture outer variables), so a DAG like `walk -> ok` becomes
    // `let ok = .. in letrec walk = .. in body` — exactly the working shape.
    //
    // PATTERN bindings in the group (`(th', tb') = case … of …` alongside
    // recursive local functions — pandoc's `toLegacyTable`) previously forced
    // the whole group onto the naive nesting fallback, where forward
    // references to later siblings stub. Desugar each pattern binding into a
    // fresh whole-value binding (`$pb = rhs`) plus one Var binding per binder
    // (`th' = case $pb of (a, _) -> a`), turning the group all-Var so the
    // same SCC machinery applies.
    let desugared_storage: Vec<hir::Binding>;
    let bindings: &[hir::Binding] = if bindings.len() > 1
        && bindings.iter().any(|b| !matches!(b.pat, hir::Pat::Var(..)))
        && bindings
            .iter()
            .all(|b| matches!(b.pat, hir::Pat::Var(..)) || !b.pat.bound_vars().is_empty())
    {
        desugared_storage = desugar_pattern_bindings(ctx, bindings);
        &desugared_storage
    } else {
        bindings
    };
    if bindings.len() > 1 && bindings.iter().all(|b| matches!(b.pat, hir::Pat::Var(..))) {
        let names: Vec<Symbol> = bindings
            .iter()
            .filter_map(|b| match &b.pat {
                hir::Pat::Var(name, ..) => Some(*name),
                _ => None,
            })
            .collect();

        // Lower each RHS once and record which siblings it references.
        let mut pairs: Vec<(Var, Box<core::Expr>)> = Vec::with_capacity(bindings.len());
        let mut deps: Vec<Vec<usize>> = Vec::with_capacity(bindings.len());
        for b in bindings {
            let (name, def_id) = match &b.pat {
                hir::Pat::Var(name, def_id, _) => (*name, *def_id),
                _ => unreachable!("guarded by all(Pat::Var)"),
            };
            let rhs = lower_local_binding_rhs(ctx, b, def_id)?;
            let fvs = collect_free_vars(&rhs);
            let d: Vec<usize> = (0..names.len())
                .filter(|&j| fvs.contains(&names[j]))
                .collect();
            deps.push(d);
            let mut var = ctx.lookup_var(def_id).cloned().unwrap_or_else(|| Var {
                name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            });
            if matches!(var.ty, Ty::Error) {
                if let Some(t) = local_binding_ty(ctx, b, def_id) {
                    var.ty = t;
                }
            }
            pairs.push((var, Box::new(rhs)));
        }

        // Tarjan SCCs of the dependency graph (edge i -> j means binding i
        // references binding j). Output order is reverse-topological: a
        // component is emitted after the components it depends on, so we wrap
        // the body in reverse of that order to bind dependencies outermost.
        let sccs = strongly_connected_components(&deps);

        let mut pairs_opt: Vec<Option<(Var, Box<core::Expr>)>> =
            pairs.into_iter().map(Some).collect();
        let mut result = body;
        for scc in sccs.iter().rev() {
            if scc.len() == 1 {
                let i = scc[0];
                let (var, rhs) = pairs_opt[i].take().expect("each binding used once");
                // Self-recursive singleton -> Rec, otherwise NonRec.
                let bind = if deps[i].contains(&i) {
                    Bind::Rec(vec![(var, rhs)])
                } else {
                    Bind::NonRec(var, rhs)
                };
                result = core::Expr::Let(Box::new(bind), Box::new(result), span);
            } else {
                // Genuine mutual cycle: one recursive group.
                let group: Vec<(Var, Box<core::Expr>)> = scc
                    .iter()
                    .map(|&i| pairs_opt[i].take().expect("each binding used once"))
                    .collect();
                result = core::Expr::Let(Box::new(Bind::Rec(group)), Box::new(result), span);
            }
        }
        return Ok(result);
    }

    // Fallback: original per-binding reverse nesting (handles complex patterns).
    let mut result = body;
    for binding in bindings.iter().rev() {
        result = lower_single_let_binding(ctx, binding, result, span)?;
    }

    Ok(result)
}

/// Tarjan's strongly-connected-components on a dependency graph given as
/// adjacency lists (`adj[i]` lists the nodes `i` points to). Returns the SCCs
/// in reverse-topological order: if component A depends on component B, B
/// appears before A. Each returned inner vec lists a component's node indices.
fn strongly_connected_components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut index_of: Vec<Option<usize>> = vec![None; n];
    let mut lowlink: Vec<usize> = vec![0; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut next_index: usize = 0;
    let mut sccs: Vec<Vec<usize>> = Vec::new();

    // Iterative DFS to avoid stack overflow on large groups. Each frame tracks
    // the node and how far through its adjacency list we've progressed.
    for start in 0..n {
        if index_of[start].is_some() {
            continue;
        }
        let mut call_stack: Vec<(usize, usize)> = vec![(start, 0)];
        while let Some(&(v, i)) = call_stack.last() {
            if i == 0 {
                index_of[v] = Some(next_index);
                lowlink[v] = next_index;
                next_index += 1;
                stack.push(v);
                on_stack[v] = true;
            }
            if i < adj[v].len() {
                let w = adj[v][i];
                call_stack.last_mut().unwrap().1 += 1;
                match index_of[w] {
                    None => call_stack.push((w, 0)),
                    Some(w_idx) => {
                        if on_stack[w] {
                            lowlink[v] = lowlink[v].min(w_idx);
                        }
                    }
                }
            } else {
                // Done with v's successors; propagate lowlink to parent.
                if lowlink[v] == index_of[v].unwrap() {
                    let mut component = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(component);
                }
                call_stack.pop();
                if let Some(&(parent, _)) = call_stack.last() {
                    lowlink[parent] = lowlink[parent].min(lowlink[v]);
                }
            }
        }
    }

    sccs
}

/// The type of a local binding's RIGHT-HAND SIDE, for spreading over a pattern
/// binding's components. See `local_binding_ty`.
pub(crate) fn binding_rhs_ty(ctx: &LowerContext, binding: &hir::Binding) -> Option<Ty> {
    binding
        .sig
        .as_ref()
        .map(|s| s.ty.clone())
        .or_else(|| ctx.resolved_expr_ty_opt(binding.rhs.span()))
        .filter(|t| !matches!(t, Ty::Error))
}

/// The declared or inferred type of a LOCAL binding, for the Core `Var` that
/// binds it.
///
/// A `let`-bound variable used to carry `Ty::Error`, and codegen reads that
/// type to decide what a comparison MEANS: `let nm = takeWhile … in any (\l ->
/// l == nm) names` compared two heap ADDRESSES and was always False, while the
/// same code with `nm` a top-level signed binding worked. Both operands being
/// untyped is what sends `==` down the scalar path.
pub(crate) fn local_binding_ty(
    ctx: &LowerContext,
    binding: &hir::Binding,
    def_id: DefId,
) -> Option<Ty> {
    binding
        .sig
        .as_ref()
        .map(|s| s.ty.clone())
        .or_else(|| ctx.lookup_scheme(def_id).map(|s| s.ty.clone()))
        .or_else(|| ctx.resolved_expr_ty_opt(binding.rhs.span()))
        .filter(|t| !matches!(t, Ty::Error))
}

/// Lower a single let binding.
/// For simple variable patterns, creates a let binding.
/// For complex patterns, creates a case expression.
/// Lower a local (`let`/`where`) binding's right-hand side as a VALUE at the
/// binding's own type.
///
/// Plain `lower_expr` skips dictionary application, so a constrained value
/// bound locally — `let s = anyChar in …`, or parsec's
/// `manyTill p end = scan where scan = …` — stays an undicted closure still
/// awaiting its `Stream` dictionary. Consumers then read that closure's header
/// as the value's own, and every later argument shifts by one. Top-level
/// bindings already get this treatment via `lower_value_def`; local ones did
/// not.
///
/// The type comes from the binding's signature when it has one, otherwise from
/// its recorded scheme or the RHS's resolved occurrence type. With no type
/// available `lower_value_arg` falls back to plain lowering, so this is a
/// superset of the previous behaviour.
fn lower_local_binding_rhs(
    ctx: &mut LowerContext,
    binding: &hir::Binding,
    def_id: DefId,
) -> LowerResult<core::Expr> {
    let mut expected: Option<Ty> = binding
        .sig
        .as_ref()
        .map(|s| s.ty.clone())
        .or_else(|| ctx.lookup_scheme(def_id).map(|s| s.ty.clone()))
        .or_else(|| ctx.resolved_expr_ty_opt(binding.rhs.span()));

    // A local binding's recorded occurrence type usually keeps the monad
    // parameters open — `let s = anyChar` records `ParsecT ?s ?u ?m Char`,
    // because nothing in the binding itself pins them. The ENCLOSING
    // binding's signature does, so match the occurrence against that
    // signature's result and parameter types and substitute what they pin.
    // Without this the constraint stays `Stream ?s ?m Char`, resolves to
    // nothing, and the value is emitted still awaiting its dictionary.
    if let Some(ty) = expected.as_mut() {
        if has_type_variables(ty) {
            if let Some(sig) = ctx.current_binding_sig().cloned() {
                let mut candidates = Vec::new();
                let mut cur = &sig;
                while let Ty::Fun(param, ret) = cur {
                    candidates.push(param.as_ref().clone());
                    cur = ret.as_ref();
                }
                candidates.push(cur.clone());
                for cand in candidates {
                    if !has_type_variables(ty) {
                        break;
                    }
                    let mut subst = bhc_types::Subst::new();
                    match_ty(ty, &cand, &mut subst);
                    *ty = subst.apply(ty);
                }
            }
        }
    }

    // Lower the RHS under the binding's own type, the way `lower_value_def`
    // does for a top-level binding. Occurrences *inside* the RHS (the
    // `anyChar` in `scan = do { x <- anyChar; … }`) repair their own
    // unresolved types against this signature; without it only the enclosing
    // definition's signature is visible and the inner constraint stays open.
    let saved_sig = ctx.set_current_binding_sig(expected.clone());
    let lowered = lower_value_arg(ctx, &binding.rhs, expected.as_ref());
    ctx.restore_current_binding_sig(saved_sig);
    lowered
}

fn lower_single_let_binding(
    ctx: &mut LowerContext,
    binding: &hir::Binding,
    body: core::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    use crate::binding::collect_free_vars;

    match &binding.pat {
        // Simple variable pattern: let x = e in body
        hir::Pat::Var(name, def_id, _) => {
            let rhs = lower_local_binding_rhs(ctx, binding, *def_id)?;
            let mut var = ctx.lookup_var(*def_id).cloned().unwrap_or_else(|| Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            });
            if matches!(var.ty, Ty::Error) {
                if let Some(t) = local_binding_ty(ctx, binding, *def_id) {
                    var.ty = t;
                }
            }

            // Check if the binding is self-recursive
            let free_vars = collect_free_vars(&rhs);
            let is_recursive = free_vars.contains(name);

            let bind = if is_recursive {
                Bind::Rec(vec![(var, Box::new(rhs))])
            } else {
                Bind::NonRec(var, Box::new(rhs))
            };

            Ok(core::Expr::Let(Box::new(bind), Box::new(body), span))
        }

        // Complex pattern: let pat = e in body -> case e of pat -> body
        _ => {
            let scrutinee = lower_expr(ctx, &binding.rhs)?;
            let alt = lower_pat_to_alt(ctx, &binding.pat, body, span)?;
            Ok(core::Expr::Case(
                Box::new(scrutinee),
                vec![alt],
                Ty::Error,
                span,
            ))
        }
    }
}

/// Lower a case expression to Core.
fn lower_case(
    ctx: &mut LowerContext,
    scrutinee: &hir::Expr,
    alts: &[hir::CaseAlt],
    span: Span,
) -> LowerResult<core::Expr> {
    use crate::pattern::{bind_pattern_vars, lower_pat_to_alt_with_fallthrough};

    let scrutinee_core = lower_expr(ctx, scrutinee)?;
    if std::env::var("BHC_DBG_CASE").is_ok() {
        let pats: Vec<String> = alts.iter().map(|a| format!("{:?}", a.pat)).collect();
        let joined = pats.join(" | ");
        eprintln!(
            "[case] span={}..{} alts={} pats={}",
            span.lo.0,
            span.hi.0,
            alts.len(),
            &joined[..joined.len().min(200)]
        );
    }

    // Check if any alternative has a nested/complex sub-pattern that needs
    // fallthrough support (e.g., `Lit 0` where the literal match may fail).
    let needs_fallthrough = alts.iter().any(|alt| has_complex_subpatterns(&alt.pat));

    if !needs_fallthrough {
        // Simple case: no nested patterns, use the fast path
        let mut core_alts = Vec::with_capacity(alts.len());
        for alt in alts {
            bind_pattern_vars(ctx, &alt.pat, None);

            // For existential constructors, push a dict scope and register
            // dictionary variables so method calls in the RHS can resolve them.
            let existential_classes = get_existential_classes(ctx, &alt.pat);
            if !existential_classes.is_empty() {
                ctx.push_dict_scope();
                let mut dict_binders = Vec::new();
                for class_name in &existential_classes {
                    let dict_var =
                        ctx.fresh_var(&format!("$dict_{}", class_name.as_str()), Ty::Error, span);
                    ctx.register_dict(*class_name, dict_var.clone());
                    dict_binders.push(dict_var);
                }
                // Store for pattern lowering to reuse as alt binders
                ctx.existential_dict_binders = dict_binders;
            }

            let rhs = if alt.guards.is_empty() {
                lower_expr(ctx, &alt.rhs)?
            } else {
                lower_guarded_rhs(ctx, &alt.guards, &alt.rhs, span)?
            };

            if !existential_classes.is_empty() {
                ctx.pop_dict_scope();
            }

            let core_alt = lower_pat_to_alt(ctx, &alt.pat, rhs, span)?;
            core_alts.push(core_alt);
        }
        return Ok(core::Expr::Case(
            Box::new(scrutinee_core),
            core_alts,
            Ty::Error,
            span,
        ));
    }

    // Complex case: some alternatives have nested sub-patterns.
    // Bind the scrutinee to a variable so fallthrough can re-case on it.
    let scrut_var = ctx.fresh_var("scrut", Ty::Error, span);

    // First, lower all alternatives' RHS and patterns (we need them all
    // to build fallthrough expressions).
    let mut lowered_alts: Vec<(hir::Pat, core::Expr)> = Vec::with_capacity(alts.len());
    for alt in alts {
        bind_pattern_vars(ctx, &alt.pat, None);

        let existential_classes = get_existential_classes(ctx, &alt.pat);
        if !existential_classes.is_empty() {
            ctx.push_dict_scope();
            let mut dict_binders = Vec::new();
            for class_name in &existential_classes {
                let dict_var =
                    ctx.fresh_var(&format!("$dict_{}", class_name.as_str()), Ty::Error, span);
                ctx.register_dict(*class_name, dict_var.clone());
                dict_binders.push(dict_var);
            }
            ctx.existential_dict_binders = dict_binders;
        }

        let rhs = if alt.guards.is_empty() {
            lower_expr(ctx, &alt.rhs)?
        } else {
            lower_guarded_rhs(ctx, &alt.guards, &alt.rhs, span)?
        };
        if !existential_classes.is_empty() {
            ctx.pop_dict_scope();
        }

        lowered_alts.push((alt.pat.clone(), rhs));
    }

    // Build the core alternatives with SHARED fallthroughs. Inlining the
    // remaining alternatives into each alternative's fallthrough is
    // EXPONENTIAL: alternative 0's fallthrough contains alternative 1's,
    // which contains alternative 2's, and so on (Text.Pandoc.Builder's
    // compile hung; Readers.Docx.Symbols overflowed the stack). Instead,
    // bind a resume point per alternative —
    //   $fallthru_i = case scrut of { alt_i; _ -> $fallthru_{i+1} }
    // — and reference it by VARIABLE. Each alternative body appears at most
    // twice (outer dispatch + its resume point), keeping the total linear.
    let n = lowered_alts.len();
    let mut resume_vars: Vec<Option<_>> = (0..=n).map(|_| None).collect();
    let mut resume_binds: Vec<core::Bind> = Vec::new(); // pushed f_{n-1} first
    let mut outer_alts_rev: Vec<core::Alt> = Vec::with_capacity(n);

    for i in (0..n).rev() {
        let (ref pat, ref rhs) = lowered_alts[i];
        let fallthrough = resume_vars[i + 1]
            .as_ref()
            .map(|v: &core::Var| core::Expr::Var(v.clone(), span));
        let core_alt = lower_pat_to_alt_with_fallthrough(ctx, pat, rhs.clone(), span, fallthrough)?;

        // Resume point used by alternatives BEFORE i: try alt i, else resume
        // at i+1 (or fail the match).
        if i > 0 {
            let default_rhs = match &resume_vars[i + 1] {
                Some(v) => core::Expr::Var(v.clone(), span),
                None => make_pattern_error(span),
            };
            let resume_case = core::Expr::Case(
                Box::new(core::Expr::Var(scrut_var.clone(), span)),
                vec![
                    core_alt.clone(),
                    core::Alt {
                        con: core::AltCon::Default,
                        binders: vec![],
                        rhs: default_rhs,
                    },
                ],
                Ty::Error,
                span,
            );
            let f = ctx.fresh_var(&format!("$fallthru{i}"), Ty::Error, span);
            resume_binds.push(core::Bind::NonRec(f.clone(), Box::new(resume_case)));
            resume_vars[i] = Some(f);
        }

        outer_alts_rev.push(core_alt);
    }
    outer_alts_rev.reverse();

    let mut case_expr = core::Expr::Case(
        Box::new(core::Expr::Var(scrut_var.clone(), span)),
        outer_alts_rev,
        Ty::Error,
        span,
    );
    // Wrap the resume bindings around the dispatch case: f_1 innermost,
    // f_{n-1} outermost so each f_i sees f_{i+1} in scope.
    for bind in resume_binds.into_iter().rev() {
        case_expr = core::Expr::Let(Box::new(bind), Box::new(case_expr), span);
    }

    let bind = core::Bind::NonRec(scrut_var, Box::new(scrutinee_core));
    Ok(core::Expr::Let(Box::new(bind), Box::new(case_expr), span))
}

/// Get the existential class names for a constructor pattern.
/// Returns empty vec for non-existential constructors.
fn get_existential_classes(ctx: &LowerContext, pat: &hir::Pat) -> Vec<Symbol> {
    match pat {
        hir::Pat::Con(def_ref, _, _) | hir::Pat::RecordCon(def_ref, _, _) => {
            if let Some(info) = ctx.lookup_constructor(def_ref.def_id) {
                if info.existential_dict_count > 0 {
                    return info.existential_classes.clone();
                }
            }
            vec![]
        }
        _ => vec![],
    }
}

/// Check if a pattern has complex (non-variable, non-wildcard) sub-patterns
/// within a constructor pattern. These require fallthrough support.
fn has_complex_subpatterns(pat: &hir::Pat) -> bool {
    match pat {
        hir::Pat::Con(_, sub_pats, _) => sub_pats
            .iter()
            .any(|p| !matches!(p, hir::Pat::Var(..) | hir::Pat::Wild(_))),
        hir::Pat::RecordCon(_, fields, _) => fields
            .iter()
            .any(|fp| !matches!(&fp.pat, hir::Pat::Var(..) | hir::Pat::Wild(_))),
        _ => false,
    }
}

/// Lower guarded RHS to nested if expressions.
fn lower_guarded_rhs(
    ctx: &mut LowerContext,
    guards: &[hir::Guard],
    rhs: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    let rhs_core = lower_expr(ctx, rhs)?;

    // Build nested ifs from right to left
    let mut result = make_pattern_error(span); // Default if no guard matches

    for guard in guards.iter().rev() {
        let cond = lower_expr(ctx, &guard.cond)?;
        result = make_if_expr(cond, rhs_core.clone(), result, span);
    }

    Ok(result)
}

/// Lower an if expression to a case on Bool.
fn lower_if(
    ctx: &mut LowerContext,
    cond: &hir::Expr,
    then_br: &hir::Expr,
    else_br: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    let cond_core = lower_expr(ctx, cond)?;
    let then_core = lower_expr(ctx, then_br)?;
    let else_core = lower_expr(ctx, else_br)?;

    Ok(make_if_expr(cond_core, then_core, else_core, span))
}

/// Create a Core if expression (case on Bool).
fn make_if_expr(
    cond: core::Expr,
    then_br: core::Expr,
    else_br: core::Expr,
    span: Span,
) -> core::Expr {
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

    let true_alt = Alt {
        con: AltCon::DataCon(true_con),
        binders: vec![],
        rhs: then_br,
    };

    let false_alt = Alt {
        con: AltCon::DataCon(false_con),
        binders: vec![],
        rhs: else_br,
    };

    core::Expr::Case(Box::new(cond), vec![true_alt, false_alt], Ty::Error, span)
}

/// Lower a tuple expression to Core.
fn lower_tuple(ctx: &mut LowerContext, elems: &[hir::Expr], span: Span) -> LowerResult<core::Expr> {
    if elems.is_empty() {
        // Unit: ()
        let unit_var = Var {
            name: Symbol::intern("()"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        return Ok(core::Expr::Var(unit_var, span));
    }

    // Build tuple constructor application
    let tuple_name = Symbol::intern(&format!("({})", ",".repeat(elems.len() - 1)));
    let tuple_var = Var {
        name: tuple_name,
        id: VarId::new(0),
        ty: Ty::Error,
    };

    let mut result = core::Expr::Var(tuple_var, span);

    for elem in elems {
        let elem_core = lower_expr(ctx, elem)?;
        result = core::Expr::App(Box::new(result), Box::new(elem_core), span);
    }

    Ok(result)
}

/// Lower a list expression to Core.
fn lower_list(ctx: &mut LowerContext, elems: &[hir::Expr], span: Span) -> LowerResult<core::Expr> {
    // Build list from right to left: [a,b,c] = a : (b : (c : []))
    let nil_var = Var {
        name: Symbol::intern("[]"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let cons_var = Var {
        name: Symbol::intern(":"),
        id: VarId::new(0),
        ty: Ty::Error,
    };

    let mut result = core::Expr::Var(nil_var, span);

    for elem in elems.iter().rev() {
        let elem_core = lower_expr(ctx, elem)?;
        // Apply (:) to elem and result
        let cons_app = core::Expr::App(
            Box::new(core::Expr::Var(cons_var.clone(), span)),
            Box::new(elem_core),
            span,
        );
        result = core::Expr::App(Box::new(cons_app), Box::new(result), span);
    }

    Ok(result)
}

/// Lower a record construction to Core.
fn lower_record(
    ctx: &mut LowerContext,
    con_ref: &DefRef,
    fields: &[hir::FieldExpr],
    span: Span,
) -> LowerResult<core::Expr> {
    // Record construction becomes constructor application
    // The fields must be in the correct order for the constructor
    let con_core = lower_con(ctx, con_ref)?;

    let mut result = con_core;
    for field in fields {
        let value_core = lower_expr(ctx, &field.value)?;
        result = core::Expr::App(Box::new(result), Box::new(value_core), span);
    }

    Ok(result)
}

/// Lower field access to Core.
///
/// Field access `r.field` is compiled to a case expression that extracts
/// the appropriate field from the constructor.
fn lower_field_access(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    field: Symbol,
    span: Span,
) -> LowerResult<core::Expr> {
    let expr_core = lower_expr(ctx, expr)?;

    // Try to find the field selector information (clone to avoid borrow issues)
    let field_info = ctx.lookup_field_selector(field).cloned();

    if let Some(info) = field_info {
        // Generate a case expression to extract the field
        // case r of Con x0 x1 ... xn -> xi (where xi is the field we want)

        // Create binder variables for all fields
        let mut binders = Vec::with_capacity(info.total_fields);
        let mut result_var = None;

        for i in 0..info.total_fields {
            let var_name = format!("$field_{i}");
            let var = ctx.fresh_var(&var_name, Ty::Error, span);
            if i == info.field_index {
                result_var = Some(var.clone());
            }
            binders.push(var);
        }

        let result_var = result_var.unwrap_or_else(|| {
            // Shouldn't happen, but handle it gracefully
            binders
                .first()
                .cloned()
                .unwrap_or_else(|| ctx.fresh_var("$error", Ty::Error, span))
        });

        // Look up constructor info for the data constructor
        let con_info = ctx.lookup_constructor(info.con_id).cloned();
        let (con_name, tycon, tag) = if let Some(ci) = con_info {
            (ci.name, TyCon::new(ci.type_name, Kind::Star), ci.tag)
        } else {
            (info.con_name, TyCon::new(info.type_name, Kind::Star), 0)
        };

        let data_con = core::DataCon {
            name: con_name,
            ty_con: tycon,
            tag,
            arity: info.total_fields as u32,
        };

        // Create the case alternative
        let alt = core::Alt {
            con: core::AltCon::DataCon(data_con),
            binders,
            rhs: core::Expr::Var(result_var, span),
        };

        // Add a default case for safety
        let default_alt = core::Alt {
            con: core::AltCon::Default,
            binders: vec![],
            rhs: make_pattern_error(span),
        };

        Ok(core::Expr::Case(
            Box::new(expr_core),
            vec![alt, default_alt],
            Ty::Error,
            span,
        ))
    } else {
        // Fallback: use selector function (works for imported types where we don't have full info)
        let selector_var = Var {
            name: field,
            id: VarId::new(0),
            ty: Ty::Error,
        };

        Ok(core::Expr::App(
            Box::new(core::Expr::Var(selector_var, span)),
            Box::new(expr_core),
            span,
        ))
    }
}

/// Lower record update to Core.
///
/// Record update `r { field1 = e1, field2 = e2 }` is compiled to a case expression
/// that extracts all fields, applies updates, and reconstructs the record.
fn lower_record_update(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    fields: &[hir::FieldExpr],
    span: Span,
) -> LowerResult<core::Expr> {
    if fields.is_empty() {
        // No fields to update - just return the original expression
        return lower_expr(ctx, expr);
    }

    let expr_core = lower_expr(ctx, expr)?;

    // Try to find the field selector information for the first field (clone to avoid borrow issues)
    let first_field = &fields[0];
    let field_info = ctx.lookup_field_selector(first_field.name).cloned();
    if std::env::var("BHC_DBG_RECUPD").is_ok() {
        eprintln!(
            "[recupd] span={}..{} field={} info={}",
            span.lo.0,
            span.hi.0,
            first_field.name.as_str(),
            field_info.is_some()
        );
    }

    if let Some(info) = field_info {
        // Build a map of field name -> new value expression
        let mut updates: std::collections::HashMap<Symbol, core::Expr> =
            std::collections::HashMap::new();
        for field in fields {
            let value_core = lower_expr(ctx, &field.value)?;
            updates.insert(field.name, value_core);
        }

        // Create binder variables for all fields
        let mut binders = Vec::with_capacity(info.total_fields);
        for i in 0..info.total_fields {
            let var_name = format!("$old_{i}");
            let var = ctx.fresh_var(&var_name, Ty::Error, span);
            binders.push(var);
        }

        // Look up constructor info (clone to avoid borrow issues)
        let con_info = ctx.lookup_constructor(info.con_id).cloned();
        let field_names: Vec<Symbol> = con_info
            .as_ref()
            .map(|ci| ci.field_names.clone())
            .unwrap_or_default();

        let (con_name, tycon, tag, arity) = if let Some(ref ci) = con_info {
            (
                ci.name,
                TyCon::new(ci.type_name, Kind::Star),
                ci.tag,
                ci.arity,
            )
        } else {
            (
                info.con_name,
                TyCon::new(info.type_name, Kind::Star),
                0,
                info.total_fields as u32,
            )
        };

        // Build the constructor application with updated fields
        let data_con = core::DataCon {
            name: con_name,
            ty_con: tycon.clone(),
            tag,
            arity,
        };

        let con_var = Var {
            name: con_name,
            id: VarId::new(0),
            ty: Ty::Error,
        };
        let mut result = core::Expr::Var(con_var, span);

        // Apply each field (using updated value if present, otherwise old value)
        for (i, binder) in binders.iter().enumerate() {
            let field_name = field_names.get(i).copied();
            let field_value = if let Some(fname) = field_name {
                if let Some(new_val) = updates.get(&fname) {
                    new_val.clone()
                } else {
                    core::Expr::Var(binder.clone(), span)
                }
            } else {
                core::Expr::Var(binder.clone(), span)
            };

            result = core::Expr::App(Box::new(result), Box::new(field_value), span);
        }

        // Create the case alternative
        let alt = core::Alt {
            con: core::AltCon::DataCon(data_con),
            binders,
            rhs: result,
        };

        // Add a default case for safety
        let default_alt = core::Alt {
            con: core::AltCon::Default,
            binders: vec![],
            rhs: make_pattern_error(span),
        };

        Ok(core::Expr::Case(
            Box::new(expr_core),
            vec![alt, default_alt],
            Ty::Error,
            span,
        ))
    } else {
        // No field info available — the record type is an external STUB
        // (e.g. TagSoup's `renderOptions{ optMinimize = .. }` in
        // Text.Pandoc.Shared). Failing the whole module here blocks every
        // dependent; lower to a runtime trap instead. The code path cannot
        // work until the real library is compiled — at which point its
        // interface provides the field layout and this branch is not taken.
        Ok(make_pattern_error(span))
    }
}

/// Create a pattern match error expression.
fn make_pattern_error(span: Span) -> core::Expr {
    let error_var = Var {
        name: Symbol::intern("error"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let msg = core::Expr::Lit(
        Literal::String(Symbol::intern(&format!(
            "Non-exhaustive patterns at bytes {}..{}",
            span.lo.0, span.hi.0
        ))),
        Ty::Error,
        span,
    );
    core::Expr::App(
        Box::new(core::Expr::Var(error_var, span)),
        Box::new(msg),
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_literal() {
        let lit = Lit::Int(42);
        let result = lower_lit(&lit, Span::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_lower_tuple() {
        let mut ctx = LowerContext::new();
        let elems = vec![
            hir::Expr::Lit(Lit::Int(1), Span::default()),
            hir::Expr::Lit(Lit::Int(2), Span::default()),
        ];
        let result = lower_tuple(&mut ctx, &elems, Span::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_lower_list() {
        let mut ctx = LowerContext::new();
        let elems = vec![
            hir::Expr::Lit(Lit::Int(1), Span::default()),
            hir::Expr::Lit(Lit::Int(2), Span::default()),
        ];
        let result = lower_list(&mut ctx, &elems, Span::default());
        assert!(result.is_ok());
    }
}
