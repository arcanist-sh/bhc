//! Regression: `mapM`/`forM` are polymorphic in the Traversable container —
//! `mapM :: (Traversable t, Monad m) => (a -> m b) -> t a -> m (t b)` — not
//! pinned to `[]`.
//!
//! The builtins were schemed `(a -> m b) -> [a] -> m [b]`, forcing the
//! container to a list. `mapM` over a `Map` then failed: e.g.
//! `Text.Pandoc.Writers.Shared`'s `Context <$> mapM .. metamap` (metamap a
//! `Map Text MetaValue`) produced `m [b]`, so `Context`'s `Map Text (Val a)`
//! argument saw `expected (Map Text), found []`. The container is now a
//! quantified var of kind `* -> *`; list uses still unify (`t = []`).

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
        "expected `mapM` over a Map (and a list) to type-check, got {result:?}"
    );
}

#[test]
fn mapm_traverses_map_and_list() {
    // `mapM Just` over a `Map k v` must yield `Maybe (Map k v)`; over `[v]` it
    // must still yield `Maybe [v]`.
    check_ok(concat!(
        "module M where\n",
        "import Data.Map (Map)\n",
        "f :: Map Int Int -> Maybe (Map Int Int)\n",
        "f = mapM Just\n",
        "g :: [Int] -> Maybe [Int]\n",
        "g = mapM Just\n",
    ));
}
