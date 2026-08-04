//! Regression: a TYPE APPLICATION inside a view pattern's expression must
//! parse — `entity (TR.decimal @Integer -> Right (x, "")) = ...` (Readers.Pod).
//!
//! Two paths, both of which used to fail the parse and silently DROP the whole
//! enclosing binding (surfacing only as "unbound variable" at use sites):
//! - unqualified `f @T -> pat`: the Ident arm folds `@` into an as-pattern, so
//!   `pat_to_expr` must rebuild `Pat::As(f, T)` as the application `f @T`;
//! - qualified `M.f @T -> pat`: the QualIdent arm leaves the `@` pending, so
//!   `parse_pattern_or_view` must consume it before the view arrow.

use bhc_parser::parse_module;
use bhc_span::FileId;

fn parse(src: &str) -> (usize, usize) {
    let (module, diagnostics) = parse_module(src, FileId::new(0));
    let errors = diagnostics
        .iter()
        .filter(|d| d.severity == bhc_diagnostics::Severity::Error)
        .count();
    let fun_binds = module
        .as_ref()
        .map(|m| {
            m.decls
                .iter()
                .filter(|d| matches!(d, bhc_ast::Decl::FunBind(_)))
                .count()
        })
        .unwrap_or(0);
    (fun_binds, errors)
}

#[test]
fn qualified_view_expr_with_type_application() {
    let (fun_binds, errors) = parse(concat!(
        "module M where\n",
        "entity (TR.decimal @Integer -> Right (x, \"\")) = Just x\n",
        "entity _ = Nothing\n",
    ));
    assert_eq!(errors, 0, "no parse errors expected");
    assert_eq!(fun_binds, 1, "entity must not be dropped");
}

#[test]
fn unqualified_view_expr_with_type_application() {
    let (fun_binds, errors) = parse(concat!(
        "module M where\n",
        "entity (decode @Integer -> Just x) = Just x\n",
        "entity _ = Nothing\n",
    ));
    assert_eq!(errors, 0, "no parse errors expected");
    assert_eq!(fun_binds, 1, "entity must not be dropped");
}

#[test]
fn view_expr_with_compound_type_argument() {
    let (fun_binds, errors) = parse(concat!(
        "module M where\n",
        "f (decode @(Maybe Int) -> Just x) = x\n",
        "f _ = 0\n",
    ));
    assert_eq!(errors, 0, "no parse errors expected");
    assert_eq!(fun_binds, 1, "f must not be dropped");
}

#[test]
fn composed_view_expression_parses() {
    // `(normalise . unEscapeString -> path)` — Readers.EPUB's
    // findEntryByPathE. `.` lexes as TokenKind::Dot (not Operator), so the
    // operator-continuation branch must accept both.
    let (fun_binds, errors) = parse(concat!(
        "module M where\n",
        "f (normalise . unEscapeString -> path) a = path\n",
    ));
    assert_eq!(errors, 0, "no parse errors expected");
    assert_eq!(fun_binds, 1, "f must not be dropped");
}

#[test]
fn as_pattern_over_composed_view_parses() {
    // `e@(stripNamespace . elName -> field)` — Readers.EPUB's parseMetaItem.
    let (fun_binds, errors) = parse(concat!(
        "module M where\n",
        "parseMetaItem e@(stripNamespace . elName -> field) meta = meta\n",
    ));
    assert_eq!(errors, 0, "no parse errors expected");
    assert_eq!(fun_binds, 1, "parseMetaItem must not be dropped");
}

#[test]
fn composed_view_in_tuple_pattern_parses() {
    // `(root, T.unpack . escapeURI . T.pack -> filename) = ...` — a where
    // pattern binding whose SECOND tuple element is a composed view pattern.
    let (fun_binds, errors) = parse(concat!(
        "module M where\n",
        "g p = root ++ filename\n",
        "  where\n",
        "    (root, T.unpack . escapeURI . T.pack -> filename) = splitFileName p\n",
    ));
    assert_eq!(errors, 0, "no parse errors expected");
    assert_eq!(fun_binds, 1, "g must not be dropped");
}

#[test]
fn plain_as_pattern_still_parses() {
    let (fun_binds, errors) = parse(concat!(
        "module M where\n",
        "f key@(Just k) = key\n",
        "f Nothing = Nothing\n",
    ));
    assert_eq!(errors, 0, "no parse errors expected");
    assert_eq!(fun_binds, 1, "f must not be dropped");
}
