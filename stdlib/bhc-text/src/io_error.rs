//! Raising an `IOException` from the text library.
//!
//! Exceptions belong to the runtime, which this crate does not link (its
//! symbols would be duplicated in each static library). A file operation
//! here that failed used to return as if it had worked: `readFile` on a
//! missing file gave the empty text, `writeFile` to a directory that does
//! not exist wrote nothing and said nothing.
//!
//! The compiled program hands this crate the runtime's function for raising
//! an `IOException` once, at startup ([`bhc_text_set_io_error_hook`]), and
//! the file operations call it through [`raise`].

use std::ffi::{c_char, CString};
use std::io;
use std::sync::atomic::{AtomicPtr, Ordering};

/// The runtime's `bhc_throw_io_error_code`: raise an `IOException` for
/// `operation` on `path` that failed with the given OS error code, and
/// return what a runtime function returns while an exception is pending.
pub type Hook = unsafe extern "C" fn(*const c_char, *const c_char, i32) -> *mut u8;

static HOOK: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Tell the text library how to raise an `IOException`. Called once by the
/// compiled program's `main`, before any Haskell code runs.
#[no_mangle]
pub extern "C" fn bhc_text_set_io_error_hook(hook: Hook) {
    HOOK.store(hook as *mut (), Ordering::Release);
}

/// Raise the `IOException` of `operation` on `path`, which failed with
/// `error`. Returns the value to return to compiled code, null if no hook
/// was set (a program not started through `main`, such as a unit test).
pub(crate) fn raise(path: &str, operation: &str, error: &io::Error) -> *mut u8 {
    let hook = HOOK.load(Ordering::Acquire);
    if hook.is_null() {
        return std::ptr::null_mut();
    }
    let (Ok(path), Ok(operation)) = (CString::new(path), CString::new(operation)) else {
        return std::ptr::null_mut();
    };
    // SAFETY: `hook` was stored from a `Hook` by `bhc_text_set_io_error_hook`,
    // and both strings outlive the call.
    unsafe {
        let hook: Hook = std::mem::transmute(hook);
        hook(
            path.as_ptr(),
            operation.as_ptr(),
            error.raw_os_error().unwrap_or(0),
        )
    }
}
