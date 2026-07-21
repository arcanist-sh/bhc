//! Regression: record construction of the external `QName` type must type-check.
//!
//! `Text.Pandoc.XML.Light.QName` is an external record type that bhc stubs by
//! name. The generic constructor fallback gave it a bare fresh-var scheme with
//! no field definitions, so record syntax
//! `QName{ qName = .., qURI = .., qPrefix = .. }` inferred as a partial function
//! (`t -> t`) rather than `QName`, breaking `Text.Pandoc.Writers.OOXML`.
//! Positional construction `QName a b c` worked (permissive fresh-var
//! unification). Both forms must type-check.

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
        "expected QName program to type-check, got {result:?}"
    );
}

#[test]
fn qname_record_construction_typechecks() {
    check_ok(concat!(
        "{-# LANGUAGE OverloadedStrings #-}\n",
        "module M where\n",
        "import Data.Text (Text)\n",
        "import Text.Pandoc.XML.Light\n",
        "recn :: Text -> QName\n",
        "recn s = QName{ qName = s, qURI = Nothing, qPrefix = Nothing }\n",
    ));
}

#[test]
fn qname_positional_construction_typechecks() {
    check_ok(concat!(
        "{-# LANGUAGE OverloadedStrings #-}\n",
        "module M where\n",
        "import Data.Text (Text)\n",
        "import Text.Pandoc.XML.Light\n",
        "posn :: Text -> QName\n",
        "posn s = QName s Nothing Nothing\n",
    ));
}
