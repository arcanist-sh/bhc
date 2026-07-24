//! Regression: a `qualified` import must not leak the module's *typed* scheme
//! into the unqualified namespace.
//!
//! `register_standard_module_exports` (bhc-lower) registered a bare stub for
//! every export even when the import was `qualified`, and for a module with
//! typed signatures (`has_typed_sigs`, e.g. `Data.Text`) that bare stub carried
//! the concrete typed name. So `import qualified Data.Text as T` made bare
//! `count` resolve to `Data.Text.count :: Text -> Text -> Int`, shadowing
//! `Text.Parsec.count :: Int -> ParsecT s u m a -> ParsecT s u m [a]`. A
//! parser's repeat count then unified to `Text`, its `n + 1` demanded
//! `Num Text`, and modules like `Text.Pandoc.Readers.Creole`/`Roff`/`Mdoc`/
//! `LaTeX.Table` failed with `No instance for Num Text`.
//!
//! The bare name is still registered (some code and the non-typed-sigs
//! qualified-alias resolution rely on it existing), but for a qualified import
//! it now carries the permissive unqualified scheme (fresh vars), so it cannot
//! shadow a real binding with a wrong concrete type.

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
        "qualified `Data.Text` import must not force a bare numeric-literal use \
         through `Data.Text.count`'s Text scheme, got {result:?}"
    );
}

#[test]
fn qualified_data_text_does_not_force_num_text() {
    // Bare `count 3 …` must not resolve to `Data.Text.count :: Text -> …`
    // (which would demand `Num Text` for the literal `3`). Before the fix this
    // failed with `No instance for Num Text`.
    check_ok(concat!(
        "module M where\n",
        "import qualified Data.Text as T\n",
        "h :: [Char]\n",
        "h = count 3 (replicate 3 (toEnum 120))\n",
    ));
}
