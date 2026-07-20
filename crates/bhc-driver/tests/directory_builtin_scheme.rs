//! Regression: `System.Directory.getModificationTime` must type as
//! `FilePath -> IO UTCTime`, not `Map String a -> Maybe a`.
//!
//! Like the `optional` case, the builtin `getModificationTime` (a
//! `DefKind::Value`) resolved to a DefId whose typeck scheme is assigned by
//! index from a list that has drifted relative to bhc-lower's `builtin_funcs`,
//! so it picked up an unrelated `Map String a -> Maybe a` scheme. That made
//! `mtime <- getModificationTime fp; FileInfo{ infoFileMTime = mtime }` fail
//! (e.g. `Text.Pandoc.Class.PandocPure`). This checks it type-checks.

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
        "expected `getModificationTime` program to type-check, got {result:?}"
    );
}

#[test]
fn get_modification_time_typechecks() {
    check_ok(concat!(
        "module M where\n",
        "import System.Directory (getModificationTime)\n",
        "useMTime :: FilePath -> IO ()\n",
        "useMTime fp = do\n",
        "    _ <- getModificationTime fp\n",
        "    return ()\n",
    ));
}
