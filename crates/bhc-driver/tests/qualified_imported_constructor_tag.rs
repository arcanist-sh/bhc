//! Regression: a constructor reached only through a qualified import
//! (`import qualified Producer as P` then `P.Done`) must keep its interface
//! tag when matched in a module compiled against a package DB.
//!
//! Constructor metadata for Core lowering was looked up by the constructor's
//! bare name, which a qualified-only import never binds. The pattern then got
//! a made-up tag, so `case r of { P.Done x -> ..; _ -> .. }` always took the
//! default branch (pandoc's `base64DataURI` on attoparsec's `A.Done`).

use bhc_driver::CompilerBuilder;
use camino::Utf8PathBuf;

#[test]
fn qualified_constructor_pattern_uses_interface_tag() {
    let src_dir = tempfile::tempdir().unwrap();
    let odir = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let dump = tempfile::tempdir().unwrap();

    std::fs::write(
        src_dir.path().join("Producer.hs"),
        concat!(
            "module Producer (Res (..), run) where\n",
            "data Res = Fail Int | Partial Int | Done Int\n",
            "run :: Int -> Res\n",
            "run n = Done n\n",
        ),
    )
    .unwrap();

    let db_path = Utf8PathBuf::from(db.path().to_str().unwrap());
    let producer = CompilerBuilder::new()
        .compile_only(true)
        .odir(Utf8PathBuf::from(odir.path().to_str().unwrap()))
        .hidir(db_path.clone())
        .build()
        .unwrap();
    producer
        .compile_module_only(Utf8PathBuf::from(
            src_dir.path().join("Producer.hs").to_str().unwrap(),
        ))
        .expect("producer should compile");

    let consumer_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        consumer_dir.path().join("Consumer.hs"),
        concat!(
            "module Consumer (isDone) where\n",
            "import qualified Producer as P\n",
            "isDone :: Int -> Int\n",
            "isDone n = case P.run n of\n",
            "  P.Done x -> x\n",
            "  _ -> 0\n",
        ),
    )
    .unwrap();

    // The test binary holds only this test, so setting the variable cannot
    // race another test.
    std::env::set_var("BHC_DUMP_LLVM", dump.path());
    let consumer = CompilerBuilder::new()
        .compile_only(true)
        .odir(Utf8PathBuf::from(odir.path().to_str().unwrap()))
        .hidir(db_path.clone())
        .package_db(db_path)
        .build()
        .unwrap();
    consumer
        .compile_module_only(Utf8PathBuf::from(
            consumer_dir.path().join("Consumer.hs").to_str().unwrap(),
        ))
        .expect("consumer should compile");
    std::env::remove_var("BHC_DUMP_LLVM");

    let ir = std::fs::read_to_string(dump.path().join("Consumer.ll")).expect("IR dump");
    let switch_start = ir.find("switch i64 %tag").expect("a tag switch in isDone");
    let switch = &ir[switch_start..switch_start + ir[switch_start..].find(']').unwrap()];
    assert!(
        switch.contains("i64 2, label"),
        "`P.Done` must switch on its interface tag 2, got: {switch}"
    );
}
