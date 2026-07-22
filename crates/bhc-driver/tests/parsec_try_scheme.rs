//! Regression: the builtin `try` must not be pinned to `Control.Exception.try`
//! (`IO a -> IO (Either e a)`).
//!
//! `try` is overloaded — `Control.Exception.try` vs `Text.Parsec.try`
//! (`ParsecT s u m a -> ParsecT s u m a`). A fixed `IO`-returning scheme poisons
//! every Parsec module that uses `try`: it forces the parser's monad to `IO` and
//! injects an `Either`, so e.g. `Text.Pandoc.CSV`'s `escaped` failed with
//! "expected `Parsec Text ()`, found `IO`" and "expected `Char`, found
//! `Either …`". The scheme is now permissive (`a -> b`) so `try` unifies to
//! whichever context uses it. This program pins a non-`IO` result through `try`,
//! which the old scheme rejected.

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
        "expected `try` program to type-check, got {result:?}"
    );
}

#[test]
fn try_preserves_non_io_result_type() {
    // Under the old `IO a -> IO (Either e a)` scheme, `try x` here would be
    // `IO (Either e Int)`, clashing with the `Maybe Int` result.
    check_ok(concat!(
        "module M where\n",
        "f :: Maybe Int -> Maybe Int\n",
        "f x = try x\n",
    ));
}

#[test]
fn fail_is_monad_polymorphic() {
    // Same bug class as `try`: the curated `fail` handler pinned `String -> IO a`,
    // forcing any do-block using `fail`/`Prelude.fail` into IO (e.g. a Parsec
    // `romanNumeral` ending in `Prelude.fail "…"`). `fail` must stay
    // `MonadFail m => String -> m a`; here it must unify to `Maybe`.
    check_ok(concat!(
        "module M where\n",
        "g :: Bool -> Maybe Int\n",
        "g b = if b then fail \"no\" else return 1\n",
    ));
}
