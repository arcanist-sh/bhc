//! Regression: `throwIO`/`throw` must accept ANY exception value, not only
//! `String`.
//!
//! `throwIO :: Exception e => e -> IO a`. BHC's curated handler (context.rs,
//! `"throwIO" | "throw"`) was "simplified" to `String -> IO a`, so throwing a
//! user exception — e.g. `E.throwIO (PandocFilterError fText msg)` in
//! `Text.Pandoc.Filter.JSON` — failed with `expected [Char], found PandocError`.
//! The exception argument is now a fresh type variable (matching the ops-table
//! scheme). Same class as the earlier `try`/`fail` curated-handler fixes.

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
        "expected `throwIO` of a non-String exception to type-check, got {result:?}"
    );
}

#[test]
fn throwio_accepts_non_string_exception() {
    // Before the fix this failed with `expected [Char], found MyErr`.
    check_ok(concat!(
        "module M where\n",
        "import qualified Control.Exception as E\n",
        "data MyErr = MyErr Int\n",
        "f :: IO a\n",
        "f = E.throwIO (MyErr 5)\n",
    ));
}
