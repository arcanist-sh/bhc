//! In module `Main`, `main` must be an `IO` action. Under the closure IO model
//! the entry point runs it, so `main = 42` segfaulted at run time; GHC rejects
//! it at compile time, and so does bhc now. Only plainly non-action types are
//! rejected, so a polymorphic `main` or a type alias still passes.

use bhc_driver::Compiler;
use std::io::Write;

fn check(source: &str) -> bool {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("create temp file");
    file.write_all(source.as_bytes()).expect("write source");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8 path");
    Compiler::with_defaults()
        .expect("compiler")
        .check_file(path)
        .is_ok()
}

#[test]
fn numeric_main_is_rejected() {
    assert!(!check("module Main where\nmain = 42\n"));
    assert!(!check("module Main where\nmain :: Int\nmain = 42\n"));
}

#[test]
fn string_and_function_main_are_rejected() {
    assert!(!check("module Main where\nmain = \"hello\"\n"));
    assert!(!check(
        "module Main where\nmain :: Int -> Int\nmain x = x\n"
    ));
}

#[test]
fn actions_pass() {
    assert!(check("module Main where\nmain :: IO ()\nmain = pure ()\n"));
    assert!(check("module Main where\nmain = return ()\n"));
    assert!(check(
        "module Main where\ntype Act = IO ()\nmain :: Act\nmain = pure ()\n"
    ));
}

#[test]
fn another_module_may_define_any_main() {
    assert!(check("module Lib where\nmain :: Int\nmain = 42\n"));
}
