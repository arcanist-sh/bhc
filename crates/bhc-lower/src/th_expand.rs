//! Minimal Template Haskell declaration-splice expansion.
//!
//! This is a deliberately small slice of TH: instead of evaluating the splice
//! body in a compile-time `Q` interpreter (which bhc does not have), it
//! *recognizes* a whitelist of well-known derivers applied to a quoted type name
//! and synthesizes the declarations they would produce, using the module's own
//! `data`/`newtype` declarations for constructor and field information.
//!
//! Currently recognized:
//! - `$(makeLenses ''T)` — for each record field `_foo` of `T`, generate a
//!   van Laarhoven lens `foo` (needs only `Functor`/`fmap`, no `lens` package):
//!
//!   ```text
//!   foo k (C x0 .. xN) = fmap (\y -> C x0 .. y .. xN) (k xi)
//!   ```
//!
//! The name quotes `''T` / `'x` are parsed transparently to `Con`/`Var`, so the
//! deriver's argument is matched here as an ordinary constructor/variable.

use bhc_ast::{Clause, ConFields, Decl, Expr, FieldDecl, FunBind, Pat, Rhs};
use bhc_intern::{Ident, Symbol};
use bhc_span::Span;

/// Expand recognized TH declaration splices in `decls`, returning a new decl
/// list with each recognized `$(deriver ...)` replaced by the declarations it
/// generates. Unrecognized splices are dropped (matching the prior "ignored"
/// behavior). If there are no splices, the input is returned unchanged.
#[must_use]
pub fn expand_th_splices(decls: Vec<Decl>) -> Vec<Decl> {
    if !decls.iter().any(|d| matches!(d, Decl::Splice(_, _))) {
        return decls;
    }
    let mut out: Vec<Decl> = Vec::with_capacity(decls.len());
    for decl in &decls {
        match decl {
            Decl::Splice(body, _) => {
                if let Some(generated) = try_expand_splice(body, &decls) {
                    out.extend(generated);
                }
                // Unrecognized splice: drop it (no-op).
            }
            other => out.push(other.clone()),
        }
    }
    out
}

/// Decompose a curried application into `(head, args)` left-to-right.
fn app_spine(e: &Expr) -> (&Expr, Vec<&Expr>) {
    let mut args = Vec::new();
    let mut cur = e;
    while let Expr::App(f, x, _) = cur {
        args.push(x.as_ref());
        cur = f.as_ref();
    }
    args.reverse();
    (cur, args)
}

/// The quoted type name in a deriver argument (`''T` parses to `Con T`).
fn as_type_name(e: &Expr) -> Option<Symbol> {
    match e {
        Expr::Con(id, _) => Some(id.name),
        _ => None,
    }
}

/// Try to expand one splice body. Returns the generated declarations, or `None`
/// if the deriver / type is not recognized.
fn try_expand_splice(body: &Expr, decls: &[Decl]) -> Option<Vec<Decl>> {
    let (head, args) = app_spine(body);
    let Expr::Var(deriver, _) = head else {
        return None;
    };
    match deriver.name.as_str() {
        // makeLenses ''T   (the sole argument is the quoted type name)
        "makeLenses" | "makeLensesFor" if !args.is_empty() => {
            let ty = as_type_name(*args.last()?)?;
            let (con, fields) = find_record_con(decls, ty)?;
            Some(gen_lenses(con, fields))
        }
        _ => None,
    }
}

/// Find the first record constructor of the named `data`/`newtype`, returning
/// `(constructor name, fields)`.
fn find_record_con(decls: &[Decl], ty: Symbol) -> Option<(Symbol, &[FieldDecl])> {
    for decl in decls {
        if let Decl::DataDecl(dd) = decl {
            if dd.name.name != ty {
                continue;
            }
            for con in &dd.constrs {
                if let ConFields::Record(fields) = &con.fields {
                    return Some((con.name.name, fields.as_slice()));
                }
            }
        }
    }
    None
}

fn sp() -> Span {
    Span::default()
}
fn var_e(name: &str) -> Expr {
    Expr::Var(Ident::new(Symbol::intern(name)), sp())
}
fn con_e(name: Symbol) -> Expr {
    Expr::Con(Ident::new(name), sp())
}
fn app_e(f: Expr, x: Expr) -> Expr {
    Expr::App(Box::new(f), Box::new(x), sp())
}
fn var_p(name: &str) -> Pat {
    Pat::Var(Ident::new(Symbol::intern(name)), sp())
}

/// Generate a van Laarhoven lens `FunBind` per record field whose name starts
/// with `_` (the `makeLenses` convention: `_foo` → lens `foo`).
fn gen_lenses(con: Symbol, fields: &[FieldDecl]) -> Vec<Decl> {
    let n = fields.len();
    let binders: Vec<String> = (0..n).map(|i| format!("x{i}")).collect();
    let mut decls = Vec::new();

    for (i, field) in fields.iter().enumerate() {
        let fname = field.name.name.as_str();
        let Some(lens) = fname.strip_prefix('_') else {
            continue; // only underscore-prefixed fields get lenses
        };

        // Pattern: con (C x0 x1 .. x_{n-1})
        let con_pat = Pat::Con(
            Ident::new(con),
            binders.iter().map(|b| var_p(b)).collect(),
            sp(),
        );

        // Rebuilt constructor: C x0 .. y(at i) .. x_{n-1}
        let mut rebuilt = con_e(con);
        for (j, b) in binders.iter().enumerate() {
            let arg = if j == i { var_e("y") } else { var_e(b) };
            rebuilt = app_e(rebuilt, arg);
        }
        // \y -> rebuilt
        let setter = Expr::Lam(vec![var_p("y")], Box::new(rebuilt), sp());
        // k x_i
        let got = app_e(var_e("k"), var_e(&binders[i]));
        // fmap (\y -> ...) (k x_i)
        let rhs_expr = app_e(app_e(var_e("fmap"), setter), got);

        let clause = Clause {
            pats: vec![var_p("k"), con_pat],
            rhs: Rhs::Simple(rhs_expr, sp()),
            wheres: Vec::new(),
            span: sp(),
        };
        decls.push(Decl::FunBind(FunBind {
            doc: None,
            name: Ident::new(Symbol::intern(lens)),
            clauses: vec![clause],
            span: sp(),
        }));
    }
    decls
}
