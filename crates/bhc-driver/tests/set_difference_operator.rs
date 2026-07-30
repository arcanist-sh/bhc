//! Regression: the `(\\)` difference operator works on `Data.Set` sets, not
//! only lists.
//!
//! `(\\)` was schemed `[a] -> [a] -> [a]` (Data.List). A module importing it
//! from `Data.Set` — e.g. `Text.Pandoc.Readers.JATS`'s
//! `blocktags = S.fromList (..) \\ S.fromList canBeInline` — then failed with
//! `expected [], found Set`. It is now `a -> a -> a`, shared by both.

use bhc_driver::Compiler;
use std::io::Write;

fn check_ok(source: &str) {
    let mut file = tempfile::Builder::new().suffix(".hs").tempfile().expect("temp");
    file.write_all(source.as_bytes()).expect("write");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8");
    let compiler = Compiler::with_defaults().expect("compiler");
    assert!(
        compiler.check_file(path).is_ok(),
        "expected (\\\\) to work on Sets and lists"
    );
}

#[test]
fn set_difference_and_list_difference() {
    check_ok(concat!(
        "{-# LANGUAGE OverloadedStrings #-}\n",
        "module M where\n",
        "import qualified Data.Set as S\n",
        "import Data.Set (Set, (\\\\))\n",
        "import Data.Text (Text)\n",
        "setDiff :: [Text] -> [Text] -> Set Text\n",
        "setDiff xs ys = S.fromList xs \\\\ S.fromList ys\n",
        "listDiff :: [Int] -> [Int] -> [Int]\n",
        "listDiff xs ys = xs \\\\ ys\n",
    ));
}
