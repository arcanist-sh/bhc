//! Regression: `Parsec` is the parsec monad specialized to `Identity` —
//! `type Parsec s u = ParsecT s u Identity` — and must unify with the
//! corresponding `ParsecT s u Identity a`.
//!
//! Without the alias, `Parsec Sources st a` stayed a distinct head, so
//! `Text.Pandoc.Parsing.General`'s `readWith p t inp = runIdentity $
//! readWithM p t inp` failed (`expected Parsec, found (ParsecT Sources)`) —
//! `readWithM` produces `ParsecT s st Identity a` at `m = Identity`, which the
//! `Parsec`-typed argument would not match.

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
        "expected `Parsec s u a` to unify with `ParsecT s u Identity a`, got {result:?}"
    );
}

#[test]
fn parsec_unifies_with_parsect_identity() {
    // A `Parsec`-typed value passed to a `ParsecT _ _ Identity`-consuming
    // function only type-checks if `Parsec` expands to `ParsecT .. Identity`.
    check_ok(concat!(
        "module M where\n",
        "import Text.Parsec (ParsecT, Parsec)\n",
        "import Data.Functor.Identity (Identity)\n",
        "runP :: ParsecT s u Identity a -> a\n",
        "runP = undefined\n",
        "useP :: Parsec s u a -> a\n",
        "useP p = runP p\n",
    ));
}
