//! System libraries for BHC
//!
//! This crate provides OS interaction primitives for the BHC runtime:
//!
//! - [`io`] - File handles and buffered I/O
//! - [`environment`] - Environment variables and program arguments
//! - [`filepath`] - Path manipulation utilities
//! - [`directory`] - Directory operations
//! - [`exit`] - Program exit codes
//! - [`process`] - Process spawning and management
//!
//! # Overview
//!
//! The system libraries provide a safe, cross-platform interface to
//! operating system functionality. All operations that can fail return
//! `Result` types with descriptive errors.
//!
//! # Example
//!
//! ```no_run
//! use bhc_system::{environment, filepath, io};
//!
//! // Get program arguments
//! let args = environment::get_args();
//!
//! // Build a file path
//! let path = filepath::join(&["home", "user", "data.txt"]);
//!
//! // Read file contents
//! let contents = io::read_file(&path).unwrap();
//! ```

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod directory;
pub mod environment;
pub mod exit;
pub mod filepath;
pub mod io;
pub mod process;

// Re-export commonly used items
pub use directory::{
    create_directory, create_directory_all, current_directory, exists, is_directory, is_file,
    list_directory, remove_directory, remove_directory_all, remove_file, rename, set_current_directory,
};
pub use environment::{get_args, get_env, get_environment, lookup_env, set_env, unset_env};
pub use exit::{exit, exit_failure, exit_success, ExitCode};
pub use filepath::{
    extension, file_name, is_absolute, is_relative, join, normalize, parent, set_extension,
    split_extension, stem,
};
pub use io::{
    read_file, read_file_bytes, write_file, write_file_bytes, append_file, Handle, OpenMode,
};
pub use process::{spawn, Command, Process, ProcessOutput};
