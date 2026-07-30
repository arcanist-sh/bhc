//! Regression: a `Text.XML.Light` `Content` constructor (`Elem`, `CRef`) may be
//! used as BOTH a pattern and an expression within the same clause.
//!
//! These constructors had a curated scheme only in the by-name builtin match,
//! which reaches expression uses; the DefId-keyed registration fell through to
//! the arity fallback and produced a *different* (wrong-arity) scheme. Using
//! `Elem` as a pattern and as a mapped function in one clause — as in
//! `Text.Pandoc.Readers.Docx.Parse`'s
//! `unwrapContent (Elem element) = map Elem (unwrapElement element)` — then
//! unified the two schemes into an infinite type
//! (`t occurs in (Element -> t)`), which polluted the whole module. Both
//! registration paths now carry the same scheme.

use bhc_driver::Compiler;
use std::io::Write;

fn check_ok(source: &str) {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("create temp file");
    file.write_all(source.as_bytes()).expect("write source");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8 path");

    let compiler = Compiler::with_defaults().expect("compiler");
    let result = compiler.check_file(path);
    assert!(
        result.is_ok(),
        "expected `Elem` usable as both pattern and expression, got {result:?}"
    );
}

#[test]
fn content_constructor_as_pattern_and_expression() {
    // Mirrors Docx.Parse's unwrapContent/unwrapElement: `Elem` matched in the
    // pattern and mapped as a function in the body of the same clause.
    check_ok(concat!(
        "module M where\n",
        "import Text.Pandoc.XML.Light (Element, Content(..))\n",
        "unwrapElement :: Element -> [Element]\n",
        "unwrapElement element = [element]\n",
        "unwrapContent :: Content -> [Content]\n",
        "unwrapContent (Elem element) = map Elem (unwrapElement element)\n",
        "unwrapContent content = [content]\n",
    ));
}
