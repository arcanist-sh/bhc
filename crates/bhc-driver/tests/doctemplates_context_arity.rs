//! Regression: the `Text.DocTemplates` `Context` constructor is a one-field
//! newtype — `Context :: Map Text (Val a) -> Context a`.
//!
//! `doctemplates` is not loaded, so `Context` was stubbed by name only and the
//! arity fallback gave the constructor the wrong shape. `Context m` then stayed
//! unapplied — `expected (Context a), found (Map Text .. -> ..)` — breaking
//! every `Text.Pandoc.Writers.Shared` context helper (getField / setField /
//! defField / resetField), which pattern-match `Context m` and rebuild
//! `Context (M.insertWith .. m)`. A curated constructor scheme fixes the arity.

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
        "expected `Context` to be a one-field constructor usable in \
         getField/setField-style helpers, got {result:?}"
    );
}

#[test]
fn context_is_single_field_constructor() {
    // Mirrors Writers.Shared.defField: pattern-match `Context m`, rebuild with
    // `Context (M.insertWith .. m)`. Pre-fix the wrong arity left `Context`
    // unapplied.
    check_ok(concat!(
        "{-# LANGUAGE OverloadedStrings #-}\n",
        "module M where\n",
        "import qualified Data.Map as M\n",
        "import Data.Text (Text)\n",
        "import Text.DocTemplates (Context(..), Val(..), ToContext(..))\n",
        "defField :: Text -> a -> Context a -> Context a\n",
        "defField field val (Context m) = Context (M.insertWith f field (toVal val) m)\n",
        "  where f _newval oldval = oldval\n",
    ));
}
