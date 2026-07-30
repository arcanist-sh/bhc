//! Regression: `Text.HTML.TagSoup`'s `Tag str` constructors (`TagText`,
//! `TagOpen`, ...) may be used as BOTH a pattern and an expression in one
//! clause.
//!
//! Same split-registration hazard as the XML.Light `Content` constructors:
//! without a DefId-keyed scheme the arity fallback disagreed with the pattern
//! side. `Text.Pandoc.SelfContained`'s
//! `\case TagText s -> TagText . toText <$> ..` (matching `TagText` and rebuilding
//! it) produced an infinite type that polluted the module. A curated
//! constructor scheme (`str` polymorphic, attribute list permissive) fixes it.

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
        "expected Tag constructors usable as both pattern and expression, got {result:?}"
    );
}

#[test]
fn tag_constructors_as_pattern_and_expression() {
    check_ok(concat!(
        "{-# LANGUAGE OverloadedStrings #-}\n",
        "module M where\n",
        "import Text.HTML.TagSoup (Tag(..))\n",
        "import Data.Text (Text)\n",
        "f :: Tag Text -> Tag Text\n",
        "f (TagText s)  = TagText s\n",
        "f (TagOpen n a) = TagOpen n a\n",
        "f t = t\n",
    ));
}
