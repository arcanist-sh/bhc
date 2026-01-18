//! Desugaring pass for syntactic sugar.
//!
//! This module handles the expansion of surface syntax constructs into
//! simpler HIR forms:
//!
//! - **Do-notation**: `do { x <- e1; e2 }` -> `e1 >>= \x -> e2`
//! - **List comprehensions**: `[e | x <- xs, p]` -> `concatMap (\x -> if p then [e] else []) xs`
//! - **If expressions**: Already have HIR representation
//! - **Guards**: Converted to case expressions with boolean matching

use bhc_ast as ast;
use bhc_hir as hir;
use bhc_intern::Symbol;
use bhc_span::Span;

use crate::context::LowerContext;

/// Desugar do-notation into monadic bind and sequence operations.
///
/// ```haskell
/// do { x <- e1; e2 }
/// -- becomes --
/// e1 >>= \x -> e2
///
/// do { e1; e2 }
/// -- becomes --
/// e1 >> e2
///
/// do { let x = e1; e2 }
/// -- becomes --
/// let x = e1 in e2
/// ```
pub fn desugar_do(
    ctx: &mut LowerContext,
    stmts: &[ast::Stmt],
    span: Span,
    lower_expr: impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    if stmts.is_empty() {
        // Empty do block - this is an error, but we handle it gracefully
        return hir::Expr::Error(span);
    }

    desugar_do_stmts(ctx, stmts, span, &lower_expr, &lower_pat)
}

fn desugar_do_stmts(
    ctx: &mut LowerContext,
    stmts: &[ast::Stmt],
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    match stmts {
        [] => hir::Expr::Error(span),

        // Final statement must be an expression (Qualifier)
        [ast::Stmt::Qualifier(e, _)] => lower_expr(ctx, e),

        // Generator: x <- e
        [ast::Stmt::Generator(pat, expr, stmt_span), rest @ ..] => {
            let e = lower_expr(ctx, expr);
            let p = lower_pat(ctx, pat);
            let body = desugar_do_stmts(ctx, rest, span, lower_expr, lower_pat);

            // e >>= \p -> body
            let bind_sym = Symbol::intern(">>=");
            let bind_ref = make_var_ref(ctx, bind_sym, *stmt_span);

            let lambda = hir::Expr::Lam(vec![p], Box::new(body), span);
            let bind_app = hir::Expr::App(Box::new(bind_ref), Box::new(e), *stmt_span);
            hir::Expr::App(Box::new(bind_app), Box::new(lambda), span)
        }

        // Qualifier (not the last one): e; ...
        [ast::Stmt::Qualifier(expr, stmt_span), rest @ ..] => {
            let e = lower_expr(ctx, expr);
            let body = desugar_do_stmts(ctx, rest, span, lower_expr, lower_pat);

            // e >> body
            let seq_sym = Symbol::intern(">>");
            let seq_ref = make_var_ref(ctx, seq_sym, *stmt_span);

            let seq_app = hir::Expr::App(Box::new(seq_ref), Box::new(e), *stmt_span);
            hir::Expr::App(Box::new(seq_app), Box::new(body), span)
        }

        // Let statement: let x = e
        [ast::Stmt::LetStmt(decls, stmt_span), rest @ ..] => {
            let body = desugar_do_stmts(ctx, rest, span, lower_expr, lower_pat);
            desugar_let_decls(ctx, decls, body, *stmt_span, lower_expr, lower_pat)
        }
    }
}

/// Desugar let declarations into HIR let bindings.
fn desugar_let_decls(
    ctx: &mut LowerContext,
    decls: &[ast::Decl],
    body: hir::Expr,
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    _lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    use crate::context::DefKind;

    // First pass: bind all names
    for decl in decls {
        if let ast::Decl::FunBind(fun_bind) = decl {
            if fun_bind.clauses.len() == 1 && fun_bind.clauses[0].pats.is_empty() {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, fun_bind.name.name, DefKind::Value, fun_bind.span);
                ctx.bind_value(fun_bind.name.name, def_id);
            }
        }
    }

    // Second pass: create bindings
    let mut bindings = Vec::new();

    for decl in decls {
        if let ast::Decl::FunBind(fun_bind) = decl {
            // Simple function binding becomes a pattern binding
            if fun_bind.clauses.len() == 1 && fun_bind.clauses[0].pats.is_empty() {
                let clause = &fun_bind.clauses[0];
                let def_id = ctx.lookup_value(fun_bind.name.name)
                    .expect("do-let binding should be bound");
                let pat = hir::Pat::Var(fun_bind.name.name, def_id, fun_bind.span);
                let rhs = match &clause.rhs {
                    ast::Rhs::Simple(e, _) => lower_expr(ctx, e),
                    ast::Rhs::Guarded(guards, _) => {
                        desugar_guarded_rhs(ctx, guards, span, lower_expr)
                    }
                };

                bindings.push(hir::Binding {
                    pat,
                    sig: None,
                    rhs,
                    span: fun_bind.span,
                });
            }
        }
    }

    if bindings.is_empty() {
        body
    } else {
        hir::Expr::Let(bindings, Box::new(body), span)
    }
}

/// Desugar list comprehensions.
///
/// ```haskell
/// [e | x <- xs, p, y <- ys]
/// -- becomes --
/// concatMap (\x -> if p then concatMap (\y -> [e]) ys else []) xs
/// ```
pub fn desugar_list_comp(
    ctx: &mut LowerContext,
    expr: &ast::Expr,
    stmts: &[ast::Stmt],
    span: Span,
    lower_expr: impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    desugar_stmts_for_comp(ctx, expr, stmts, span, &lower_expr, &lower_pat)
}

fn desugar_stmts_for_comp(
    ctx: &mut LowerContext,
    expr: &ast::Expr,
    stmts: &[ast::Stmt],
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    match stmts {
        [] => {
            // [e] - singleton list
            let e = lower_expr(ctx, expr);
            hir::Expr::List(vec![e], span)
        }

        [ast::Stmt::Generator(pat, gen_expr, qual_span), rest @ ..] => {
            // x <- xs becomes concatMap (\x -> ...) xs
            let p = lower_pat(ctx, pat);
            let xs = lower_expr(ctx, gen_expr);
            let body = desugar_stmts_for_comp(ctx, expr, rest, span, lower_expr, lower_pat);

            let lambda = hir::Expr::Lam(vec![p], Box::new(body), span);

            let concat_map_sym = Symbol::intern("concatMap");
            let concat_map = make_var_ref(ctx, concat_map_sym, *qual_span);

            let app1 = hir::Expr::App(Box::new(concat_map), Box::new(lambda), *qual_span);
            hir::Expr::App(Box::new(app1), Box::new(xs), span)
        }

        [ast::Stmt::Qualifier(guard_expr, qual_span), rest @ ..] => {
            // p becomes if p then ... else []
            let cond = lower_expr(ctx, guard_expr);
            let then_branch = desugar_stmts_for_comp(ctx, expr, rest, span, lower_expr, lower_pat);
            let else_branch = hir::Expr::List(vec![], *qual_span);

            hir::Expr::If(
                Box::new(cond),
                Box::new(then_branch),
                Box::new(else_branch),
                span,
            )
        }

        [ast::Stmt::LetStmt(decls, qual_span), rest @ ..] => {
            // let x = e becomes let x = e in ...
            let body = desugar_stmts_for_comp(ctx, expr, rest, span, lower_expr, lower_pat);
            desugar_let_decls(ctx, decls, body, *qual_span, lower_expr, lower_pat)
        }
    }
}

/// Desugar guarded right-hand sides to nested if expressions.
///
/// ```haskell
/// | g1 = e1
/// | g2 = e2
/// | otherwise = e3
/// -- becomes --
/// if g1 then e1 else if g2 then e2 else e3
/// ```
pub fn desugar_guarded_rhs(
    ctx: &mut LowerContext,
    guards: &[ast::GuardedRhs],
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
) -> hir::Expr {
    guards.iter().rev().fold(
        // Default: error "Non-exhaustive guards"
        make_pattern_match_error(ctx, span),
        |else_branch, guard| {
            let cond = lower_expr(ctx, &guard.guard);
            let then_branch = lower_expr(ctx, &guard.body);
            hir::Expr::If(Box::new(cond), Box::new(then_branch), Box::new(else_branch), span)
        },
    )
}

/// Create a reference to a variable (looking it up in scope).
fn make_var_ref(ctx: &mut LowerContext, name: Symbol, span: Span) -> hir::Expr {
    if let Some(def_id) = ctx.lookup_value(name) {
        hir::Expr::Var(ctx.def_ref(def_id, span))
    } else {
        // If not found, create a placeholder (will be caught during type checking)
        let def_id = ctx.fresh_def_id();
        ctx.define(def_id, name, crate::context::DefKind::Value, span);
        hir::Expr::Var(ctx.def_ref(def_id, span))
    }
}

/// Create a pattern match failure error expression.
fn make_pattern_match_error(ctx: &mut LowerContext, span: Span) -> hir::Expr {
    let error_sym = Symbol::intern("error");
    let error_ref = make_var_ref(ctx, error_sym, span);
    let msg = hir::Expr::Lit(
        hir::Lit::String(Symbol::intern("Non-exhaustive patterns")),
        span,
    );
    hir::Expr::App(Box::new(error_ref), Box::new(msg), span)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desugar_simple_do() {
        let mut ctx = LowerContext::with_builtins();

        // do { map } -- using a builtin that's bound
        let ident = bhc_intern::Ident::from_str("map");
        let stmts = vec![ast::Stmt::Qualifier(
            ast::Expr::Var(ident, Span::default()),
            Span::default(),
        )];

        let result = desugar_do(
            &mut ctx,
            &stmts,
            Span::default(),
            |ctx, e| {
                if let ast::Expr::Var(ident, span) = e {
                    let name = ident.name;
                    if let Some(def_id) = ctx.lookup_value(name) {
                        return hir::Expr::Var(ctx.def_ref(def_id, *span));
                    }
                }
                hir::Expr::Error(Span::default())
            },
            |_ctx, _p| hir::Pat::Wild(Span::default()),
        );

        // Result should be a Var (since `map` is a builtin)
        assert!(matches!(result, hir::Expr::Var(_)));
    }
}
