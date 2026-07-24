//! Regression: `anyToken` must be resolvable through the `Text.Parsec`
//! aggregate re-export.
//!
//! `register_standard_module_exports` (bhc-lower) lists what each builtin module
//! re-exports. `anyToken` was present in the `Text.Parsec.Combinator` sub-list
//! but MISSING from the `Text.Parsec` aggregate — even though `Text.Parsec`
//! re-exports it. So `import Text.Parsec` (as `Text.Pandoc.Parsing` does) did not
//! bring `anyToken` into scope, and any module using it through that chain (e.g.
//! `Text.Pandoc.Readers.Mdoc`) failed AST->HIR lowering with
//! `unbound variable: anyToken` while sibling combinators (`manyTill`, `eof`)
//! resolved fine.

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
        "expected `anyToken` (via Text.Parsec) to resolve, got {result:?}"
    );
}

#[test]
fn any_token_resolves_through_text_parsec() {
    check_ok(concat!(
        "module M (p) where\n",
        "import Text.Parsec\n",
        "p :: Monad m => ParsecT s u m t\n",
        "p = anyToken\n",
    ));
}
