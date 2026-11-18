//! A runtime panic in a compiled program is reported as a one-line internal
//! error with a bug-report hint and status 1, not a Rust panic message and
//! an abort (#99).
//!
//! The test runs itself again as a child that starts the runtime the way a
//! compiled program does and then panics.

use std::process::Command;

const CHILD: &str = "BHC_RTS_INTERNAL_ERROR_CHILD";

#[test]
fn runtime_panic_is_reported_as_internal_error() {
    if std::env::var_os(CHILD).is_some() {
        unsafe { bhc_rts::ffi::bhc_rts_init(0, std::ptr::null()) };
        panic!("an invariant failed");
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "runtime_panic_is_reported_as_internal_error",
            "--exact",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env_remove("RUST_BACKTRACE")
        .output()
        .expect("failed to run the test binary");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "stderr:\n{stderr}");
    assert!(
        stderr.contains("internal error: an invariant failed"),
        "stderr:\n{stderr}"
    );
    assert!(stderr.contains("This is a bug in bhc"), "stderr:\n{stderr}");
    assert!(!stderr.contains("panicked at"), "stderr:\n{stderr}");
}
