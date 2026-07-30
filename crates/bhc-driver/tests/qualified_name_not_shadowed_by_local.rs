//! Regression: a qualified reference `M.name` must resolve to the imported
//! module's binding, even when the current module *locally* defines `name`.
//!
//! For a non-`has_typed_sigs` import, `M.name` was registered only as a
//! shadowable indirection to the unqualified `name` (`register_qualified_name`).
//! A local top-level/class-method definition of `name` overwrote the unqualified
//! binding, so `M.name` was hijacked by the local. In
//! `Text.Pandoc.Class.PandocMonad`, the class method `trace :: Text -> m ()`
//! shadowed `Debug.Trace.trace :: String -> a -> a`, so
//! `Debug.Trace.trace msg (return ())` applied the 1-arg method result `m ()`
//! to `(return ())` — `expected (m ()), found (m () -> t)`. The qualified alias
//! is now bound directly to the import stub, which wins during resolution.

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
        "expected qualified `Debug.Trace.trace` to resolve to the import (2-arg), \
         not the local `trace`, got {result:?}"
    );
}

#[test]
fn qualified_reference_ignores_local_same_name() {
    // Local `trace :: String -> Int` must NOT hijack `Debug.Trace.trace`
    // (`String -> a -> a`), which is applied to two arguments here.
    check_ok(concat!(
        "module M where\n",
        "import qualified Debug.Trace\n",
        "trace :: String -> Int\n",
        "trace _ = 0\n",
        "f :: Int\n",
        "f = Debug.Trace.trace \"x\" (5 :: Int)\n",
    ));
}
