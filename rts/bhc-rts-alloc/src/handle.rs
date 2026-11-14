//! A file handle, in one place.
//!
//! The runtime and the text library both read and write through handles,
//! and each kept its own copy of the struct ("must have identical layout")
//! and of the numbers that stood for the standard streams. A standard
//! stream was the number itself in a value word: 1, 2 or 3 where a pointer
//! belongs.
//!
//! A handle is a heap object like any other (`object`): word 0 is
//! [`object::HANDLE`]. Word 1 says which stream it is, [`FILE`] for an
//! opened file and [`STDIN`], [`STDOUT`] or [`STDERR`] for a standard one;
//! the rest is the file and its state.
//!
//! The standard streams are told apart by word 1, not by address: this
//! crate is linked into several static libraries, each with its own copy
//! of any static.

use std::fs::File;

use crate::object;

/// Word 1 of a handle on an opened file.
pub const FILE: usize = 0;
/// Word 1 of the standard input handle.
pub const STDIN: usize = 1;
/// Word 1 of the standard output handle.
pub const STDOUT: usize = 2;
/// Word 1 of the standard error handle.
pub const STDERR: usize = 3;

/// A handle: the kind word of a heap object, the stream it stands for, and
/// for an opened file the file and its state.
#[repr(C)]
pub struct Handle {
    kind: i64,
    stream: usize,
    /// The open file; `None` once closed, and for a standard stream.
    pub file: Option<File>,
    /// Whether the file was opened for reading.
    pub readable: bool,
    /// Whether the file was opened for writing.
    pub writable: bool,
    /// Whether `hClose` has run.
    pub closed: bool,
}

impl Handle {
    /// A handle on an opened file.
    #[must_use]
    pub const fn on_file(file: File, readable: bool, writable: bool) -> Self {
        Self {
            kind: object::HANDLE,
            stream: FILE,
            file: Some(file),
            readable,
            writable,
            closed: false,
        }
    }

    /// The handle of a standard stream: [`STDIN`], [`STDOUT`] or [`STDERR`].
    #[must_use]
    pub const fn standard(stream: usize) -> Self {
        Self {
            kind: object::HANDLE,
            stream,
            file: None,
            readable: stream == STDIN,
            writable: stream != STDIN,
            closed: false,
        }
    }
}

/// Which stream `handle` is: [`FILE`], [`STDIN`], [`STDOUT`] or [`STDERR`].
/// A null handle is no standard stream, so [`FILE`].
///
/// # Safety
///
/// `handle` must be null or point to a [`Handle`].
#[inline]
#[must_use]
pub unsafe fn stream(handle: *const u8) -> usize {
    if handle.is_null() {
        return FILE;
    }
    // SAFETY: the caller guarantees a `Handle`.
    unsafe { (*handle.cast::<Handle>()).stream }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_handle_is_a_heap_object_that_names_its_stream() {
        let stdout = Handle::standard(STDOUT);
        let pointer = std::ptr::addr_of!(stdout).cast::<u8>();
        // SAFETY: `pointer` is a live `Handle`.
        unsafe {
            assert_eq!(object::tag(pointer), object::HANDLE);
            assert_eq!(stream(pointer), STDOUT);
            assert_eq!(stream(std::ptr::null()), FILE);
        }
        assert!(object::is_first_word(object::HANDLE));
    }
}
