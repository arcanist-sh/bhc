//! Regression: the total UTF-8 decoders `decodeUtf8'` return
//! `Either UnicodeException Text`, not `Text`.
//!
//! bhc previously stubbed `Data.Text.Lazy.Encoding.decodeUtf8'` (and the
//! strict `Data.Text.Encoding.decodeUtf8'`) with the same `ByteString -> Text`
//! scheme as the throwing `decodeUtf8`. Code that pattern-matches the result —
//! `case decodeUtf8' bs of { Left _ -> ..; Right t -> .. }`, as in
//! `Text.Pandoc.PDF`'s `utf8ToText` — then forced `Text ~ Either e a`,
//! surfacing as `expected (Either e a), found Text`. Worse, the bad
//! unification polluted shared state so a *sibling* binding (`showVerboseInfo`)
//! reported the error, making it an emergent whole-module failure. The `'`
//! decoders are now typed `ByteString -> Either e Text` (error side left
//! polymorphic, since bhc does not model `UnicodeException`).

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
        "expected `case decodeUtf8' bs of Left/Right` to type-check, got {result:?}"
    );
}

#[test]
fn decode_utf8_prime_is_either_typed() {
    // The scrutinee must accept `Left`/`Right` patterns — i.e. it is
    // `Either e Text`, not `Text`.
    check_ok(concat!(
        "module M where\n",
        "import Data.ByteString.Lazy (ByteString)\n",
        "import Data.Text.Lazy.Encoding (decodeUtf8')\n",
        "f :: ByteString -> Int\n",
        "f bs = case decodeUtf8' bs of\n",
        "         Left _  -> 0\n",
        "         Right _ -> 1\n",
    ));
}
