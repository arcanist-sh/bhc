//! Regression test: a *qualified* class name in a constraint context
//! (`B.ToMetaValue a => ...`) must parse as a `Type::Constrained`, exactly like
//! the unqualified form (`ToMetaValue a => ...`).
//!
//! `try_parse_constraint` only accepted `TokenKind::ConId` for the class, so a
//! qualified class (`TokenKind::QualConId`) made it return `None`; the context
//! went unrecognized and `B.ToMetaValue a =>` leaked into the function type as
//! a value argument. Downstream, typeck then expected `ToMetaValue t` where a
//! plain function sat — e.g. `Text.Pandoc.Readers.Txt2Tags`'s
//! `headerline :: B.ToMetaValue a => Text -> T2T a -> T2T ()` failed with
//! `expected (B.ToMetaValue t), found (t -> t)`.

use bhc_ast::{Decl, Type};
use bhc_parser::parse_module;
use bhc_span::FileId;

fn parse(src: &str) -> bhc_ast::Module {
    let (module, diags) = parse_module(src, FileId::new(0));
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == bhc_diagnostics::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "unexpected parse errors: {errors:?}");
    module.expect("module should parse")
}

fn sig_ty<'a>(module: &'a bhc_ast::Module, name: &str) -> &'a Type {
    module
        .decls
        .iter()
        .find_map(|d| match d {
            Decl::TypeSig(s) if s.names.iter().any(|n| n.name.as_str() == name) => Some(&s.ty),
            _ => None,
        })
        .expect("type signature present")
}

#[test]
fn qualified_class_constraint_parses_as_context() {
    // Both the qualified and unqualified constraint forms must yield a
    // `Constrained` type whose single constraint names the class `ToMetaValue`.
    let module = parse(
        "\
module T where

import qualified Text.Pandoc.Builder as B

qual :: B.ToMetaValue a => Text -> a -> Meta
qual = undefined

unqual :: ToMetaValue a => Text -> a -> Meta
unqual = undefined
",
    );

    for name in ["qual", "unqual"] {
        match sig_ty(&module, name) {
            Type::Constrained(constraints, _inner, _) => {
                assert_eq!(constraints.len(), 1, "{name}: one constraint");
                assert_eq!(
                    constraints[0].class.name.as_str(),
                    "ToMetaValue",
                    "{name}: constraint class is ToMetaValue (qualifier dropped)"
                );
            }
            other => panic!("{name}: expected Constrained, got {other:?}"),
        }
    }
}

#[test]
fn parenthesized_qualified_constraints_parse() {
    // A parenthesized context mixing qualified and unqualified classes.
    let module = parse(
        "\
module T where

import qualified Text.Pandoc.Builder as B

f :: (B.ToMetaValue a, Show a) => a -> Text
f = undefined
",
    );
    match sig_ty(&module, "f") {
        Type::Constrained(constraints, _, _) => {
            assert_eq!(constraints.len(), 2, "two constraints");
            let names: Vec<_> = constraints.iter().map(|c| c.class.name.as_str()).collect();
            assert!(names.contains(&"ToMetaValue"), "has ToMetaValue: {names:?}");
            assert!(names.contains(&"Show"), "has Show: {names:?}");
        }
        other => panic!("expected Constrained, got {other:?}"),
    }
}
