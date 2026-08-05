//! Regression: `.bhi` emission must not be gated on codegen success — the
//! interface is typecheck-level truth, generated from the AST and typed HIR
//! only. When emission ran last, any post-typecheck failure (lowering bug,
//! LLVM verifier error, object write) left no interface, so one codegen bug
//! cost every dependent's compile instead of just this module's `.o`: in the
//! Pandoc topo sweep, MediaBag/URI verifier failures left no interfaces,
//! Class.PandocMonad then lowered against stubs and failed, and its whole
//! downstream (Class, Highlighting, ...) cascaded.

use bhc_driver::CompilerBuilder;
use camino::Utf8PathBuf;

#[test]
fn interface_survives_post_typecheck_failure() {
    let src_dir = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let db_path = Utf8PathBuf::from(db.path().to_str().unwrap());

    std::fs::write(
        src_dir.path().join("Prod.hs"),
        concat!(
            "module Prod (prod) where\n",
            "prod :: Int -> Int\n",
            "prod x = x + 1\n",
        ),
    )
    .unwrap();

    // Force a failure AFTER type checking: odir's parent is a regular file,
    // so the object-move step errors while parse/lower/typecheck all succeed
    // (standing in for a lowering or LLVM-verifier failure).
    let blocker = src_dir.path().join("not-a-dir");
    std::fs::write(&blocker, "").unwrap();
    let bad_odir = Utf8PathBuf::from(blocker.join("odir").to_str().unwrap());

    let producer = CompilerBuilder::new()
        .compile_only(true)
        .odir(bad_odir)
        .hidir(db_path.clone())
        .build()
        .unwrap();
    let result = producer.compile_module_only(Utf8PathBuf::from(
        src_dir.path().join("Prod.hs").to_str().unwrap(),
    ));
    assert!(result.is_err(), "producer should fail at the object step");
    assert!(
        db.path().join("Prod.bhi").exists(),
        "interface must be written despite the post-typecheck failure"
    );

    // Consumer compiles to a real object against that interface alone.
    let consumer_dir = tempfile::tempdir().unwrap();
    let odir = tempfile::tempdir().unwrap();
    std::fs::write(
        consumer_dir.path().join("UseProd.hs"),
        concat!(
            "module UseProd (useProd) where\n",
            "import Prod (prod)\n",
            "useProd :: Int -> Int\n",
            "useProd x = prod x + 1\n",
        ),
    )
    .unwrap();

    let consumer = CompilerBuilder::new()
        .compile_only(true)
        .odir(Utf8PathBuf::from(odir.path().to_str().unwrap()))
        .hidir(db_path.clone())
        .package_db(db_path)
        .build()
        .unwrap();
    consumer
        .compile_module_only(Utf8PathBuf::from(
            consumer_dir.path().join("UseProd.hs").to_str().unwrap(),
        ))
        .expect("consumer should compile against the failed producer's interface");

    let obj = odir.path().join("UseProd.o");
    assert!(obj.exists(), "expected object at {}", obj.display());
}
