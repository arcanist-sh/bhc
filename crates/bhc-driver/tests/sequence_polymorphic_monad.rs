//! Regression: `sequence`/`sequence_` are polymorphic in the monad
//! (`Monad m => [m a] -> m [a]`), not IO-specialized.
//!
//! `sequence` was `[IO a] -> IO [a]`, so `sequence :: [Maybe a] -> Maybe [a]`
//! — as in `Text.Pandoc.Readers.JATS`'s `case sequence ws of Just ..` over
//! `[Maybe ColWidth]` — forced `Maybe ~ IO` (`expected IO, found Maybe`).

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
        "expected sequence to work in Maybe"
    );
}

#[test]
fn sequence_in_maybe() {
    check_ok(concat!(
        "module M where\n",
        "sm :: [Maybe Int] -> Maybe [Int]\n",
        "sm = sequence\n",
    ));
}
