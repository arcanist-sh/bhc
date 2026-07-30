//! Regression: an arrow-polymorphic type `a x y` (with an `Arrow`/`ArrowChoice`
//! constraint) must unify with a plain function `x -> y` by choosing `a := (->)`,
//! and the Control.Arrow/Category operators must carry their real fixities.
//!
//! `Text.Pandoc.Readers.ODT.Arrows.Utils` defines
//!   `a >>?^ f = a >>^ Left ^|||^ Right . f`
//! where `^|||^ :: ArrowChoice a => (b -> d) -> (c -> d) -> a (Either b c) d`.
//! Two bugs kept it from checking:
//!   1. `>>^` (really `infixr 1`) had no fixity, so it defaulted to `infixl 9`
//!      and the body mis-grouped as `(a >>^ Left) ^|||^ (Right . f)`, feeding
//!      `^|||^` an arrow where it wanted a function.
//!   2. Even correctly grouped, `Left ^|||^ (Right . f)` is `a (Either ..) (..)`
//!      but `>>^`'s second argument is a plain function `(c -> d)`; the unifier
//!      had no `App`-vs-`Fun` cross-type arm, so it could not pick `a := (->)`.

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
        "expected arrow-as-function unification to check"
    );
}

#[test]
fn arrow_polymorphic_value_unifies_with_plain_function() {
    // `a (Either b c) d` used where `Either b c -> d` is expected → `a := (->)`.
    check_ok(concat!(
        "{-# LANGUAGE FlexibleContexts #-}\n",
        "module M where\n",
        "import Control.Arrow (ArrowChoice)\n",
        "myEither :: (ArrowChoice a) => (b -> d) -> (c -> d) -> a (Either b c) d\n",
        "myEither = undefined\n",
        "use :: (b -> d) -> (c -> d) -> (Either b c -> d)\n",
        "use l r = myEither l r\n",
    ));
}

#[test]
fn control_arrow_operators_have_standard_fixities() {
    // `>>^` is `infixr 1` and `^|||^` `infixr 2`, so `a >>^ l ^|||^ r` must group
    // as `a >>^ (l ^|||^ r)`. Combined with the unification fix, the ODT idiom
    // `a >>^ Left ^|||^ Right . f` type-checks.
    check_ok(concat!(
        "{-# LANGUAGE FlexibleContexts #-}\n",
        "module M where\n",
        "import Control.Arrow (ArrowChoice, (>>^), arr, (|||))\n",
        "infixr 2 ^|||^\n",
        "(^|||^) :: (ArrowChoice a) => (b -> d) -> (c -> d) -> a (Either b c) d\n",
        "l ^|||^ r = arr l ||| arr r\n",
        "pipe :: (ArrowChoice a) => a x (Either f s) -> (s -> s')\n",
        "        -> a x (Either f s')\n",
        "pipe a f = a >>^ Left ^|||^ Right . f\n",
    ));
}
