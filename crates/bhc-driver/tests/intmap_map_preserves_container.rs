//! Regression: `Data.IntMap.map`/`filter` operate on the `IntMap` container,
//! not on the element type — `map :: (a -> b) -> IntMap a -> IntMap b`.
//!
//! The schemes used the element type variable for the map position
//! (`(a -> b) -> a -> b`), so applying `IntMap.map (res:) (sRawTokens st)` in
//! `Text.Pandoc.Readers.LaTeX.Parsing`, where `sRawTokens :: IntMap [Tok]`,
//! forced that field to a bare `[Tok]` (`expected [], found IntMap`). The
//! container is now a higher-kinded variable (`f a -> f b`).

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
        "expected IntMap.map/filter to keep the IntMap container, got {result:?}"
    );
}

#[test]
fn intmap_map_and_filter_keep_container() {
    // `IntMap.map (0:)` over an `IntMap [Int]` must yield `IntMap [Int]`, not
    // force the map to be `[Int]`.
    check_ok(concat!(
        "module M where\n",
        "import qualified Data.IntMap as IntMap\n",
        "import Data.IntMap (IntMap)\n",
        "bump :: IntMap [Int] -> IntMap [Int]\n",
        "bump m = IntMap.map (0:) m\n",
        "keepBig :: IntMap Int -> IntMap Int\n",
        "keepBig = IntMap.filter (> 5)\n",
    ));
}
