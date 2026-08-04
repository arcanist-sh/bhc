//! Regression coverage for the 2026-08-04 stub/scheme gaps found on the Pandoc
//! grind. Each case mirrors the construct that failed in a real module.

use bhc_driver::Compiler;
use std::io::Write;

fn check_ok(source: &str) {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("temp");
    file.write_all(source.as_bytes()).expect("write");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8");
    let compiler = Compiler::with_defaults().expect("compiler");
    assert!(
        compiler.check_file(path).is_ok(),
        "expected the module to check"
    );
}

#[test]
fn aeson_null_is_not_pinned_to_block() {
    // `Null` had a stale curated scheme `Null :: Block` (pandoc-types <1.23);
    // a case over an Aeson `Value` scrutinee failed with `expected Value,
    // found Block` (Readers.Metadata's yamlToMetaValue).
    check_ok(concat!(
        "module M where\n",
        "import Data.Aeson (Value(..))\n",
        "describe :: Value -> String\n",
        "describe v = case v of\n",
        "  String _ -> \"string\"\n",
        "  Null     -> \"null\"\n",
        "  _        -> \"other\"\n",
    ));
}

#[test]
fn tagsoup_attribute_alias_expands_to_pair() {
    // `type Attribute str = (str, str)` (TagSoup). Opaque, it clashed with the
    // tuples `lookup` produces over attribute lists (Readers.HTML's
    // `pSelfClosing (== \"img\") (isJust . lookup \"src\")`).
    check_ok(concat!(
        "module M where\n",
        "import Data.Maybe (isJust)\n",
        "import Text.HTML.TagSoup\n",
        "import qualified Data.Text as T\n",
        "hasSrc :: [Attribute T.Text] -> Bool\n",
        "hasSrc = isJust . lookup \"src\"\n",
    ));
}

#[test]
fn seq_cons_pattern_synonym_resolves() {
    // `Seq.:<|` (Data.Sequence's pattern synonym) was missing from the stub
    // exports; Readers.HTML matches `Header ... Seq.:<| rest`.
    check_ok(concat!(
        "module M where\n",
        "import qualified Data.Sequence as Seq\n",
        "firstOf :: Seq.Seq Int -> Maybe Int\n",
        "firstOf s = case s of\n",
        "  x Seq.:<| _ -> Just x\n",
        "  _           -> Nothing\n",
    ));
}

#[test]
fn doclayout_before_non_blank_is_unary() {
    // `beforeNonBlank :: Doc a -> Doc a` shared prefixed's BINARY scheme, so
    // `beforeNonBlank \";\"` stayed a partial application (Writers.Typst's
    // `endCode :: Doc Text`).
    check_ok(concat!(
        "{-# LANGUAGE OverloadedStrings #-}\n",
        "module M where\n",
        "import Text.DocLayout\n",
        "import Data.Text (Text)\n",
        "endCode :: Doc Text\n",
        "endCode = beforeNonBlank \";\"\n",
    ));
}

#[test]
fn jira_markup_constructors_are_dedicated_stubs() {
    // Text.Jira.Markup exports collide with pandoc-types constructor names
    // (Str, Space, Citation, ...); the module is on the dedicated-stub path so
    // `Jira.Str` must not resolve to pandoc-types' `Str`.
    check_ok(concat!(
        "module M where\n",
        "import Data.Text (Text)\n",
        "import qualified Text.Jira.Markup as Jira\n",
        "toText :: Jira.Inline -> Maybe Text\n",
        "toText (Jira.Str t) = Just t\n",
        "toText Jira.Space = Nothing\n",
        "toText _ = Nothing\n",
    ));
}
