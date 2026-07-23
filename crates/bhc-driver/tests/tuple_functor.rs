//! Regression: a tuple must unify against an HKT `f a` shape, so `fmap`/`<$>`
//! over a 2-tuple type-checks (tuples are `Functor`/`Foldable`/`Traversable`
//! in their last field — `(,) c`).
//!
//! bhc represents tuples as a dedicated `Ty::Tuple` variant, so `f a` (an
//! `App`) could not unify with `(x, y)`. `Text.Pandoc.Writers.JATS.References`
//! failed on `T.dropWhile isdash <$> T.break isdash val` (`T.break` returns
//! `(Text, Text)`). The unifier now bridges `App(f, a)` and `Ty::Tuple`, the
//! same way it already bridges `App(f, a)` and `Ty::List`.

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
        "expected tuple-functor program to type-check, got {result:?}"
    );
}

#[test]
fn fmap_over_pair_typechecks() {
    // `(,) c` is a Functor over its second component: `(+1) <$> (0, 1) == (0, 2)`.
    check_ok(concat!(
        "module M where\n",
        "f :: (Int, Int)\n",
        "f = (+1) <$> (0, 1)\n",
    ));
}
