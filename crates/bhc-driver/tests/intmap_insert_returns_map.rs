//! Regression: `Data.IntMap.insert`'s curated scheme must return the *map*
//! type, not the *value* type.
//!
//! These curated IntMap schemes model the map as an opaque type variable (there
//! is no first-class `IntMap` type in the checker). `insert` was
//! `Int -> a -> a -> a`, conflating the value and the map into the same `a`, so
//! `insert k v tbl` was inferred as the value type. When the map type is pinned
//! concretely elsewhere — e.g. a `foldl'` accumulator `(Int, IntMap X)` as in
//! `Text.Pandoc.Readers.RTF`'s font/list tables — this clashed:
//! `expected (IntMap.IntMap FontFamily), found FontFamily`.
//!
//! The map now uses its own variable `b` (`insert :: Int -> a -> b -> b`), so
//! `insert k v tbl` returns `tbl`'s type.

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
        "expected IntMap.insert to return the map type, got {result:?}"
    );
}

#[test]
fn intmap_insert_into_fold_accumulator() {
    // `insert n Decor tbl` must be `IntMap FF` (== `tbl`), matching the
    // accumulator `(Int, IntMap FF)`. Before the fix this was inferred as `FF`
    // and failed with `expected IntMap FontFamily, found FontFamily`.
    check_ok(concat!(
        "module M where\n",
        "import qualified Data.IntMap as IntMap\n",
        "import Data.List (foldl')\n",
        "data FF = Decor | Tech\n",
        "build :: (Int, IntMap.IntMap FF) -> [Int] -> (Int, IntMap.IntMap FF)\n",
        "build acc ts = foldl' go acc ts\n",
        "  where go (n, tbl) _ = (n, IntMap.insert n Decor tbl)\n",
    ));
}
