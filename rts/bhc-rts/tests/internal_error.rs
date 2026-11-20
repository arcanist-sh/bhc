//! A runtime panic in a compiled program is reported as a one-line internal
//! error with a bug-report hint and status 1, not a Rust panic message and
//! an abort (#99).
//!
//! Each test runs itself again as a child that starts the runtime the way a
//! compiled program does and then panics.

use std::process::Command;

const CHILD: &str = "BHC_RTS_INTERNAL_ERROR_CHILD";

/// Run the named test as a child and check it reported `message` as an
/// internal error.
fn assert_reported_as_internal_error(test: &str, message: &str) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([test, "--exact", "--nocapture"])
        .env(CHILD, "1")
        .env_remove("RUST_BACKTRACE")
        .output()
        .expect("failed to run the test binary");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "stderr:\n{stderr}");
    assert!(
        stderr.contains(&format!("internal error: {message}")),
        "stderr:\n{stderr}"
    );
    assert!(stderr.contains("This is a bug in bhc"), "stderr:\n{stderr}");
    assert!(!stderr.contains("panicked at"), "stderr:\n{stderr}");
}

fn is_child() -> bool {
    std::env::var_os(CHILD).is_some()
}

#[test]
fn runtime_panic_is_reported_as_internal_error() {
    if is_child() {
        unsafe { bhc_rts::ffi::bhc_rts_init(0, std::ptr::null()) };
        panic!("an invariant failed");
    }
    assert_reported_as_internal_error(
        "runtime_panic_is_reported_as_internal_error",
        "an invariant failed",
    );
}

/// `main` runs on a thread of its own, and a panic inside an `extern "C"`
/// function cannot unwind: both must still be reported.
#[test]
fn panic_on_the_main_thread_is_reported_as_internal_error() {
    extern "C" fn body() {
        let n = 3;
        panic!("an invariant failed: {n}");
    }
    if is_child() {
        unsafe {
            bhc_rts::ffi::bhc_rts_init(0, std::ptr::null());
            bhc_rts::stack::bhc_run_main(body);
        }
        return;
    }
    assert_reported_as_internal_error(
        "panic_on_the_main_thread_is_reported_as_internal_error",
        "an invariant failed: 3",
    );
}
