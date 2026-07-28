//! Regression: a `let`/`where` binding may reference a tuple-pattern component
//! *before* the tuple binding, and must get the component's type — not the
//! whole tuple.
//!
//! Let-group inference (infer.rs `Expr::Let`) pre-registered EVERY variable of a
//! binding's pattern with the same whole-binding type variable. So for
//! `(pre, sp) = …`, both `pre` and `sp` took the entire tuple type, and a
//! forward reference like `h = pre` (appearing before the tuple binding)
//! resolved `pre` to `(t, t)` instead of its component. This surfaced as
//! `expected Int, found (t, t)` / `expected (t, Text), found Text` and blocked
//! `Text.Pandoc.Format`, `Text.Pandoc.Parsing.GridTable`, and
//! `Text.Pandoc.Readers.Man` (all destructure a tuple in a `where` whose
//! components feed sibling bindings).
//!
//! Each pattern variable now gets its own fresh type var; `check_pattern`
//! projects the components, so the reference order no longer matters.

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
        "expected forward reference to a tuple-pattern component to type-check, got {result:?}"
    );
}

#[test]
fn where_binding_forward_refs_tuple_component() {
    // `h = pre` appears BEFORE `(pre, sp) = (10, 20)`; `pre` must be `Int`.
    check_ok(concat!(
        "module M where\n",
        "g :: Int\n",
        "g = h\n",
        "  where\n",
        "    h = pre\n",
        "    (pre, sp) = (10, 20)\n",
    ));
}

#[test]
fn let_binding_forward_refs_tuple_component() {
    // Same, order-dependent, in a `let`.
    check_ok(concat!(
        "module M where\n",
        "g :: Int\n",
        "g = let h = sp\n",
        "        (pre, sp) = (10, 20)\n",
        "    in h\n",
    ));
}
