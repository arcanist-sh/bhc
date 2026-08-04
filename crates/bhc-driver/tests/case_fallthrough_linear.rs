//! Regression: lowering a case whose alternatives have NESTED sub-patterns
//! must stay LINEAR in the number of alternatives.
//!
//! The hir-to-core "complex case" path used to build each alternative's
//! fallthrough by cloning all later alternatives — alternative 0's tree
//! contained alternative 1's, which contained alternative 2's, giving O(2^n)
//! Core. Text.Pandoc.Builder's compile hung for over nine minutes and
//! Readers.Docx.Symbols overflowed the stack. With shared `$fallthru_i`
//! resume points the total stays linear and this test compiles in seconds;
//! before the fix, 24 alternatives (~2^24 nodes) effectively never finishes.

use bhc_driver::CompilerBuilder;
use camino::Utf8PathBuf;

#[test]
fn nested_pattern_case_with_many_alternatives_compiles() {
    let dir = tempfile::tempdir().unwrap();
    let odir = tempfile::tempdir().unwrap();
    let hidir = tempfile::tempdir().unwrap();

    // 24 alternatives, each with a nested sub-pattern (Just (Left/Right …))
    // so every alternative takes the complex-case path and needs a
    // fallthrough for inner-pattern failure.
    let mut alts = String::new();
    for i in 0..12 {
        alts.push_str(&format!(
            "  f (Just (Left {i})) = {i}\n  f (Just (Right {i})) = {i} + 100\n"
        ));
    }
    let src = format!(
        "module ManyAlts (g) where\n\
         g :: Maybe (Either Int Int) -> Int\n\
         g x = f x\n\
          where\n\
         {alts}  f _ = -1\n"
    );
    let hs = dir.path().join("ManyAlts.hs");
    std::fs::write(&hs, src).unwrap();

    let compiler = CompilerBuilder::new()
        .compile_only(true)
        .odir(Utf8PathBuf::from(odir.path().to_str().unwrap()))
        .hidir(Utf8PathBuf::from(hidir.path().to_str().unwrap()))
        .build()
        .unwrap();
    compiler
        .compile_module_only(Utf8PathBuf::from(hs.to_str().unwrap()))
        .expect("many-alternative nested-pattern case should compile");

    let obj = odir.path().join("ManyAlts.o");
    assert!(obj.exists(), "expected object at {}", obj.display());
}

#[test]
fn direct_builtin_bridge_covers_container_cafs() {
    // Container builtins reached through the DIRECT (pre-lowered argument)
    // path — e.g. `Set.fromList` inside a top-level CAF pipeline, the shape
    // that broke Text.Pandoc.Extensions — must route through the generic
    // value bridge instead of erroring "unhandled container builtin".
    let dir = tempfile::tempdir().unwrap();
    let odir = tempfile::tempdir().unwrap();
    let hidir = tempfile::tempdir().unwrap();

    let src = "module ContainerCaf (bigSet, keyed) where\n\
               import qualified Data.Set as Set\n\
               import qualified Data.Map as Map\n\
               bigSet :: Set.Set Int\n\
               bigSet = Set.union (Set.fromList [1, 2, 3]) (Set.fromList [3, 4])\n\
               keyed :: Maybe Int\n\
               keyed = Map.lookup 2 (Map.insert 2 20 (Map.fromList [(1, 10)]))\n";
    let hs = dir.path().join("ContainerCaf.hs");
    std::fs::write(&hs, src).unwrap();

    let compiler = CompilerBuilder::new()
        .compile_only(true)
        .odir(Utf8PathBuf::from(odir.path().to_str().unwrap()))
        .hidir(Utf8PathBuf::from(hidir.path().to_str().unwrap()))
        .build()
        .unwrap();
    compiler
        .compile_module_only(Utf8PathBuf::from(hs.to_str().unwrap()))
        .expect("container CAFs should compile via the direct-builtin bridge");

    let obj = odir.path().join("ContainerCaf.o");
    assert!(obj.exists(), "expected object at {}", obj.display());
}
