//! Regression: `Data.Text.foldr`'s folding function takes the Char element
//! FIRST (`(Char -> a -> a) -> a -> Text -> a`). It shared `Data.Text.foldl`'s
//! scheme (`(a -> Char -> a) -> ...`), so a `\case`-style function matching
//! Char literals had its patterns unified with the ACCUMULATOR — Readers.Org.
//! Blocks' `T.foldr (\case {'\t' -> (tabStop +); _ -> (1 +)}) 0` failed with
//! `expected Int, found Char` plus a bogus `Num Char` constraint.

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
fn text_foldr_char_element_comes_first() {
    // Mirrors Org.Blocks' countSpaces: the lambda's Char pattern must meet the
    // element position, the numeric sections the accumulator.
    check_ok(concat!(
        "{-# LANGUAGE LambdaCase #-}\n",
        "module M where\n",
        "import qualified Data.Text as T\n",
        "countSpaces :: Int -> T.Text -> Int\n",
        "countSpaces tabStop =\n",
        "  T.foldr (\\case {'\\t' -> (tabStop +); _ -> (1 +)}) 0\n",
    ));
}

#[test]
fn text_foldl_accumulator_still_comes_first() {
    // foldl keeps its own shape: accumulator first, Char second.
    check_ok(concat!(
        "module M where\n",
        "import qualified Data.Text as T\n",
        "countChars :: T.Text -> Int\n",
        "countChars = T.foldl (\\acc _ -> acc + 1) 0\n",
    ));
}
