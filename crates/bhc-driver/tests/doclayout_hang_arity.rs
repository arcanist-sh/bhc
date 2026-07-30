//! Regression: `Text.DocLayout.hang` takes THREE arguments —
//! `hang :: HasChars a => Int -> Doc a -> Doc a -> Doc a` (indent, prefix,
//! body) — not two.
//!
//! `hang` was bundled with `nest`/`cblock`/`lblock`/`rblock` under the shared
//! `Int -> a -> a` scheme, dropping its third argument. So `hang n prefix`
//! typed as a finished `Doc` instead of `Doc -> Doc`; in
//! `Text.Pandoc.Writers.ANSI`'s `number | doNumber = hang .. .. | otherwise = id`
//! the two guard branches then disagreed (`Doc` vs `Doc -> Doc`), yielding
//! `expected (Doc Text), found (a -> a)`.

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
        "expected `hang` to accept three arguments (indent, prefix, body), got {result:?}"
    );
}

#[test]
fn hang_takes_three_arguments() {
    // `hang 2 prefix doc` applies three args; pre-fix `hang` was 2-arity and
    // applying the third argument failed.
    check_ok(concat!(
        "module M where\n",
        "import qualified Text.DocLayout as D\n",
        "h prefix doc = D.hang 2 prefix doc\n",
    ));
}
