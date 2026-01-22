//! Program exit codes and termination
//!
//! This module provides functions for terminating the program with
//! specific exit codes.
//!
//! # Exit Codes
//!
//! - 0: Success
//! - 1: General failure
//! - Other values: Application-specific errors
//!
//! # Example
//!
//! ```no_run
//! use bhc_system::exit::{exit, exit_success, exit_failure, ExitCode};
//!
//! // Exit with success
//! exit_success();
//!
//! // Exit with failure
//! exit_failure();
//!
//! // Exit with custom code
//! exit(ExitCode::new(42));
//! ```

use std::process;

/// An exit code for program termination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExitCode(i32);

impl ExitCode {
    /// Success exit code (0)
    pub const SUCCESS: ExitCode = ExitCode(0);

    /// Failure exit code (1)
    pub const FAILURE: ExitCode = ExitCode(1);

    /// Create a new exit code from an integer
    ///
    /// # Example
    ///
    /// ```
    /// use bhc_system::exit::ExitCode;
    ///
    /// let code = ExitCode::new(42);
    /// assert_eq!(code.value(), 42);
    /// ```
    pub const fn new(code: i32) -> Self {
        ExitCode(code)
    }

    /// Get the integer value of the exit code
    pub const fn value(&self) -> i32 {
        self.0
    }

    /// Check if this is a success code (0)
    pub const fn is_success(&self) -> bool {
        self.0 == 0
    }

    /// Check if this is a failure code (non-zero)
    pub const fn is_failure(&self) -> bool {
        self.0 != 0
    }
}

impl Default for ExitCode {
    fn default() -> Self {
        Self::SUCCESS
    }
}

impl From<i32> for ExitCode {
    fn from(code: i32) -> Self {
        ExitCode(code)
    }
}

impl From<ExitCode> for i32 {
    fn from(code: ExitCode) -> Self {
        code.0
    }
}

impl std::fmt::Display for ExitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Exit the program with the given exit code
///
/// This function never returns.
///
/// # Example
///
/// ```no_run
/// use bhc_system::exit::{exit, ExitCode};
///
/// exit(ExitCode::SUCCESS);
/// ```
pub fn exit(code: ExitCode) -> ! {
    process::exit(code.0)
}

/// Exit the program with a success code (0)
///
/// This function never returns.
pub fn exit_success() -> ! {
    process::exit(0)
}

/// Exit the program with a failure code (1)
///
/// This function never returns.
pub fn exit_failure() -> ! {
    process::exit(1)
}

/// Exit with a custom integer code
///
/// This function never returns.
pub fn exit_with(code: i32) -> ! {
    process::exit(code)
}

/// Exit handler that can be used to register cleanup actions
///
/// Note: This is a simplified interface. In practice, use `atexit`
/// or similar mechanisms for cleanup.
pub struct ExitHandler {
    handlers: Vec<Box<dyn FnOnce() + Send + 'static>>,
}

impl ExitHandler {
    /// Create a new exit handler
    pub fn new() -> Self {
        ExitHandler {
            handlers: Vec::new(),
        }
    }

    /// Register a cleanup handler
    pub fn on_exit<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.handlers.push(Box::new(f));
    }

    /// Run all handlers and exit
    pub fn exit(self, code: ExitCode) -> ! {
        for handler in self.handlers {
            handler();
        }
        exit(code)
    }
}

impl Default for ExitHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Result type that can be converted to an exit code
pub trait Termination {
    /// Convert to an exit code
    fn report(self) -> ExitCode;
}

impl Termination for () {
    fn report(self) -> ExitCode {
        ExitCode::SUCCESS
    }
}

impl Termination for ExitCode {
    fn report(self) -> ExitCode {
        self
    }
}

impl<T, E: std::fmt::Debug> Termination for Result<T, E> {
    fn report(self) -> ExitCode {
        match self {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("Error: {:?}", e);
                ExitCode::FAILURE
            }
        }
    }
}

impl Termination for bool {
    fn report(self) -> ExitCode {
        if self {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    }
}

impl Termination for i32 {
    fn report(self) -> ExitCode {
        ExitCode::new(self)
    }
}

/// Exit with the result of a termination value
///
/// This function never returns.
pub fn exit_with_result<T: Termination>(result: T) -> ! {
    exit(result.report())
}

// FFI exports

/// Exit with code (FFI)
#[no_mangle]
pub extern "C" fn bhc_exit(code: i32) -> ! {
    exit(ExitCode::new(code))
}

/// Exit with success (FFI)
#[no_mangle]
pub extern "C" fn bhc_exit_success() -> ! {
    exit_success()
}

/// Exit with failure (FFI)
#[no_mangle]
pub extern "C" fn bhc_exit_failure() -> ! {
    exit_failure()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_code_new() {
        let code = ExitCode::new(42);
        assert_eq!(code.value(), 42);
    }

    #[test]
    fn test_exit_code_constants() {
        assert_eq!(ExitCode::SUCCESS.value(), 0);
        assert_eq!(ExitCode::FAILURE.value(), 1);
    }

    #[test]
    fn test_exit_code_is_success() {
        assert!(ExitCode::SUCCESS.is_success());
        assert!(!ExitCode::FAILURE.is_success());
        assert!(ExitCode::new(0).is_success());
        assert!(!ExitCode::new(1).is_success());
    }

    #[test]
    fn test_exit_code_is_failure() {
        assert!(!ExitCode::SUCCESS.is_failure());
        assert!(ExitCode::FAILURE.is_failure());
        assert!(!ExitCode::new(0).is_failure());
        assert!(ExitCode::new(42).is_failure());
    }

    #[test]
    fn test_exit_code_from_i32() {
        let code: ExitCode = 5.into();
        assert_eq!(code.value(), 5);
    }

    #[test]
    fn test_exit_code_to_i32() {
        let code = ExitCode::new(10);
        let value: i32 = code.into();
        assert_eq!(value, 10);
    }

    #[test]
    fn test_exit_code_display() {
        let code = ExitCode::new(42);
        assert_eq!(format!("{}", code), "42");
    }

    #[test]
    fn test_termination_unit() {
        assert!(().report().is_success());
    }

    #[test]
    fn test_termination_exit_code() {
        assert!(ExitCode::SUCCESS.report().is_success());
        assert!(ExitCode::FAILURE.report().is_failure());
    }

    #[test]
    fn test_termination_result() {
        let ok: Result<i32, &str> = Ok(42);
        assert!(ok.report().is_success());

        let err: Result<i32, &str> = Err("error");
        assert!(err.report().is_failure());
    }

    #[test]
    fn test_termination_bool() {
        assert!(true.report().is_success());
        assert!(false.report().is_failure());
    }

    #[test]
    fn test_termination_i32() {
        assert!(0.report().is_success());
        assert!(1.report().is_failure());
    }

    #[test]
    fn test_exit_handler() {
        let mut handler = ExitHandler::new();
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();

        handler.on_exit(move || {
            called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        // We can't actually test exit, but we can verify the handler is registered
        assert_eq!(handler.handlers.len(), 1);
    }
}
