//! Regression: `foldMap` must keep its Monoid result polymorphic, not pinned to
//! a list. The DefId-keyed builtin scheme used to be `(a -> [b]) -> [a] -> [b]`,
//! which unified any non-list monoid result (e.g. `foldMap fromInline` producing
//! `CslJson Text` in Text.Pandoc.Writers.CslJson) with `[]`, yielding symmetric
//! `expected [] / found CslJson` type errors. The correct scheme is
//! `(Foldable t, Monoid m) => (a -> m) -> t a -> m`.

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
fn foldmap_produces_non_list_monoid() {
    // `foldMap fromOne :: [Int] -> Acc` — the result is a user monoid, not `[]`.
    // This mirrors CslJson's `fromInlines = foldMap fromInline . B.fromList`.
    check_ok(concat!(
        "module M where\n",
        "newtype Acc = Acc [Int]\n",
        "instance Semigroup Acc where\n",
        "  Acc a <> Acc b = Acc (a ++ b)\n",
        "instance Monoid Acc where\n",
        "  mempty = Acc []\n",
        "fromOne :: Int -> Acc\n",
        "fromOne n = Acc [n]\n",
        "fromMany :: [Int] -> Acc\n",
        "fromMany = foldMap fromOne\n",
    ));
}

#[test]
fn foldmap_over_list_still_yields_list() {
    // The common list case must keep working: `foldMap` with a list-valued
    // function over a list of lists still produces a list (the container var
    // unifies with `[]`).
    check_ok(concat!(
        "module M where\n",
        "flatten :: [[Int]] -> [Int]\n",
        "flatten = foldMap id\n",
    ));
}
