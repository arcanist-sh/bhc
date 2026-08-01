//! Regression: a multi-line `if`/`then`/`else` whose `then`/`else` land at the
//! enclosing layout column must parse. Haskell 2010 allows an optional
//! semicolon before `then`/`else` (`if e [;] then e [;] else e`, the
//! DoAndIfThenElse rule); the lexer inserts a VirtualSemi before a `then`/`else`
//! sitting at the layout column, and the parser must accept it.
//!
//! Text.Pandoc.Writers.FB2's `renderSections` has
//!   let blocks'' = if null initialBlocks
//!       then blocks'
//!       else Div ... : secs
//! where `then`/`else` align with the let-bound name. Before the fix the `if`
//! parser choked on the inserted `;`, the parse error dropped the whole
//! `renderSections` binding, and importers saw it as unbound.

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
fn then_else_at_let_layout_column() {
    // `then`/`else` align with the let-bound name `z` (the FB2 shape). The
    // enclosing binding `g` must not be dropped.
    check_ok(concat!(
        "module M where\n",
        "g :: Bool -> IO Int\n",
        "g y = do\n",
        "    let z = if y\n",
        "        then 1\n",
        "        else 2\n",
        "    return z\n",
        "usesG :: Bool -> IO Int\n",
        "usesG = g\n",
    ));
}

#[test]
fn then_else_at_do_layout_column() {
    // A bare `if` statement in a `do` block with `then`/`else` at the do-layout
    // column (also inserts a VirtualSemi before them).
    check_ok(concat!(
        "module M where\n",
        "h :: Bool -> IO ()\n",
        "h y = do\n",
        "  if y\n",
        "  then putStrLn \"a\"\n",
        "  else putStrLn \"b\"\n",
        "usesH :: Bool -> IO ()\n",
        "usesH = h\n",
    ));
}

#[test]
fn ordinary_if_still_parses() {
    // Guard against over-eager semicolon eating: normal `if` (single line and
    // then/else indented under `if`) must keep working.
    check_ok(concat!(
        "module M where\n",
        "f :: Bool -> Int\n",
        "f y = if y then 1 else 2\n",
        "k :: Bool -> Int\n",
        "k y = if y\n",
        "      then 1\n",
        "      else 2\n",
    ));
}
