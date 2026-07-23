//! Regression: `Data.Tree.Node` must have its real polymorphic constructor
//! scheme `forall a. a -> [Tree a] -> Tree a`.
//!
//! `Data.Tree` is stubbed, so without a curated scheme `Node` fell through to
//! the generic fresh-result fallback (`a -> [b] -> c`); construction like
//! `Node x [] :: Tree a` then failed ("expected `Tree SI`, found `SI -> t`"),
//! which broke `Text.Pandoc.Chunks` (`toTOCTree = Node SecInfo{…} . …`).

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
        "expected Data.Tree.Node program to type-check, got {result:?}"
    );
}

#[test]
fn data_tree_node_construction_and_match() {
    check_ok(concat!(
        "module M where\n",
        "import Data.Tree (Tree(..))\n",
        "data SI = SI { field :: Int }\n",
        "g :: Tree SI\n",
        "g = Node (SI 0) []\n",
        "h :: Tree SI -> SI\n",
        "h (Node x _) = x\n",
    ));
}
