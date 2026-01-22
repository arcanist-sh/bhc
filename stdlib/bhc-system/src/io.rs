//! File I/O operations
//!
//! This module provides file handle abstractions and buffered I/O operations.
//!
//! # Overview
//!
//! The I/O system is built around the [`Handle`] type, which represents an
//! open file or stream. Handles support reading, writing, and seeking operations.
//!
//! # Example
//!
//! ```no_run
//! use bhc_system::io::{Handle, OpenMode, read_file, write_file};
//!
//! // Simple file operations
//! write_file("greeting.txt", "Hello, World!").unwrap();
//! let contents = read_file("greeting.txt").unwrap();
//!
//! // Using handles for more control
//! let mut handle = Handle::open("data.txt", OpenMode::Read).unwrap();
//! let mut buffer = vec![0u8; 1024];
//! let bytes_read = handle.read(&mut buffer).unwrap();
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Error type for I/O operations
#[derive(Debug)]
pub struct IoError {
    /// The kind of error
    pub kind: IoErrorKind,
    /// Human-readable error message
    pub message: String,
}

/// Categories of I/O errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoErrorKind {
    /// File or directory not found
    NotFound,
    /// Permission denied
    PermissionDenied,
    /// File already exists
    AlreadyExists,
    /// Invalid input or argument
    InvalidInput,
    /// Unexpected end of file
    UnexpectedEof,
    /// Operation would block
    WouldBlock,
    /// Operation interrupted
    Interrupted,
    /// Other I/O error
    Other,
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for IoError {}

impl From<io::Error> for IoError {
    fn from(err: io::Error) -> Self {
        let kind = match err.kind() {
            io::ErrorKind::NotFound => IoErrorKind::NotFound,
            io::ErrorKind::PermissionDenied => IoErrorKind::PermissionDenied,
            io::ErrorKind::AlreadyExists => IoErrorKind::AlreadyExists,
            io::ErrorKind::InvalidInput => IoErrorKind::InvalidInput,
            io::ErrorKind::UnexpectedEof => IoErrorKind::UnexpectedEof,
            io::ErrorKind::WouldBlock => IoErrorKind::WouldBlock,
            io::ErrorKind::Interrupted => IoErrorKind::Interrupted,
            _ => IoErrorKind::Other,
        };
        IoError {
            kind,
            message: err.to_string(),
        }
    }
}

/// Result type for I/O operations
pub type IoResult<T> = Result<T, IoError>;

/// File open mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    /// Open for reading only
    Read,
    /// Open for writing, truncating existing content
    Write,
    /// Open for appending to existing content
    Append,
    /// Open for both reading and writing
    ReadWrite,
}

/// Seek position for file operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekPosition {
    /// Seek from the start of the file
    Start(u64),
    /// Seek from the end of the file (negative offset)
    End(i64),
    /// Seek from the current position
    Current(i64),
}

impl From<SeekPosition> for SeekFrom {
    fn from(pos: SeekPosition) -> Self {
        match pos {
            SeekPosition::Start(n) => SeekFrom::Start(n),
            SeekPosition::End(n) => SeekFrom::End(n),
            SeekPosition::Current(n) => SeekFrom::Current(n),
        }
    }
}

/// Internal handle state
enum HandleInner {
    /// Buffered reader
    Reader(BufReader<File>),
    /// Buffered writer
    Writer(BufWriter<File>),
    /// Read-write handle (unbuffered for simplicity)
    ReadWrite(File),
}

/// A file handle for I/O operations
///
/// Handles are automatically closed when dropped. They support
/// buffered reading and writing for efficiency.
pub struct Handle {
    inner: Arc<Mutex<HandleInner>>,
    mode: OpenMode,
    path: String,
}

impl Handle {
    /// Open a file with the specified mode
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bhc_system::io::{Handle, OpenMode};
    ///
    /// let handle = Handle::open("file.txt", OpenMode::Read).unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P, mode: OpenMode) -> IoResult<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = match mode {
            OpenMode::Read => OpenOptions::new().read(true).open(&path)?,
            OpenMode::Write => OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)?,
            OpenMode::Append => OpenOptions::new()
                .write(true)
                .create(true)
                .append(true)
                .open(&path)?,
            OpenMode::ReadWrite => OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&path)?,
        };

        let inner = match mode {
            OpenMode::Read => HandleInner::Reader(BufReader::new(file)),
            OpenMode::Write | OpenMode::Append => HandleInner::Writer(BufWriter::new(file)),
            OpenMode::ReadWrite => HandleInner::ReadWrite(file),
        };

        Ok(Handle {
            inner: Arc::new(Mutex::new(inner)),
            mode,
            path: path_str,
        })
    }

    /// Get the file path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the open mode
    pub fn mode(&self) -> OpenMode {
        self.mode
    }

    /// Read bytes into a buffer
    ///
    /// Returns the number of bytes read. Returns 0 at end of file.
    pub fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            HandleInner::Reader(reader) => Ok(reader.read(buf)?),
            HandleInner::ReadWrite(file) => Ok(file.read(buf)?),
            HandleInner::Writer(_) => Err(IoError {
                kind: IoErrorKind::InvalidInput,
                message: "Cannot read from write-only handle".to_string(),
            }),
        }
    }

    /// Read all remaining bytes
    pub fn read_all(&mut self) -> IoResult<Vec<u8>> {
        let mut buf = Vec::new();
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            HandleInner::Reader(reader) => {
                reader.read_to_end(&mut buf)?;
            }
            HandleInner::ReadWrite(file) => {
                file.read_to_end(&mut buf)?;
            }
            HandleInner::Writer(_) => {
                return Err(IoError {
                    kind: IoErrorKind::InvalidInput,
                    message: "Cannot read from write-only handle".to_string(),
                });
            }
        }
        Ok(buf)
    }

    /// Read a single line (including newline)
    pub fn read_line(&mut self) -> IoResult<Option<String>> {
        let mut line = String::new();
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            HandleInner::Reader(reader) => {
                let bytes = reader.read_line(&mut line)?;
                if bytes == 0 {
                    Ok(None)
                } else {
                    Ok(Some(line))
                }
            }
            HandleInner::ReadWrite(file) => {
                // For read-write, we need to wrap temporarily
                let mut reader = BufReader::new(file);
                let bytes = reader.read_line(&mut line)?;
                if bytes == 0 {
                    Ok(None)
                } else {
                    Ok(Some(line))
                }
            }
            HandleInner::Writer(_) => Err(IoError {
                kind: IoErrorKind::InvalidInput,
                message: "Cannot read from write-only handle".to_string(),
            }),
        }
    }

    /// Write bytes to the file
    ///
    /// Returns the number of bytes written.
    pub fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            HandleInner::Writer(writer) => Ok(writer.write(buf)?),
            HandleInner::ReadWrite(file) => Ok(file.write(buf)?),
            HandleInner::Reader(_) => Err(IoError {
                kind: IoErrorKind::InvalidInput,
                message: "Cannot write to read-only handle".to_string(),
            }),
        }
    }

    /// Write all bytes to the file
    pub fn write_all(&mut self, buf: &[u8]) -> IoResult<()> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            HandleInner::Writer(writer) => Ok(writer.write_all(buf)?),
            HandleInner::ReadWrite(file) => Ok(file.write_all(buf)?),
            HandleInner::Reader(_) => Err(IoError {
                kind: IoErrorKind::InvalidInput,
                message: "Cannot write to read-only handle".to_string(),
            }),
        }
    }

    /// Flush buffered data to the underlying file
    pub fn flush(&mut self) -> IoResult<()> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            HandleInner::Writer(writer) => Ok(writer.flush()?),
            HandleInner::ReadWrite(file) => Ok(file.flush()?),
            HandleInner::Reader(_) => Ok(()), // Nothing to flush
        }
    }

    /// Seek to a position in the file
    ///
    /// Returns the new position from the start of the file.
    pub fn seek(&mut self, pos: SeekPosition) -> IoResult<u64> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            HandleInner::Reader(reader) => Ok(reader.seek(pos.into())?),
            HandleInner::Writer(writer) => Ok(writer.seek(pos.into())?),
            HandleInner::ReadWrite(file) => Ok(file.seek(pos.into())?),
        }
    }

    /// Get the current position in the file
    pub fn position(&mut self) -> IoResult<u64> {
        self.seek(SeekPosition::Current(0))
    }

    /// Check if at end of file
    pub fn is_eof(&mut self) -> IoResult<bool> {
        let mut inner = self.inner.lock().unwrap();
        match &mut *inner {
            HandleInner::Reader(reader) => {
                let buf = reader.fill_buf()?;
                Ok(buf.is_empty())
            }
            HandleInner::ReadWrite(file) => {
                let pos = file.stream_position()?;
                let end = file.seek(SeekFrom::End(0))?;
                file.seek(SeekFrom::Start(pos))?;
                Ok(pos >= end)
            }
            HandleInner::Writer(_) => Ok(false),
        }
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Handle {
            inner: Arc::clone(&self.inner),
            mode: self.mode,
            path: self.path.clone(),
        }
    }
}

// Standard handles

/// Get a handle to standard input
pub fn stdin() -> Handle {
    // Create a dummy handle for stdin - in real implementation would use actual stdin
    Handle {
        inner: Arc::new(Mutex::new(HandleInner::Reader(BufReader::new(
            // This is a placeholder - real impl would use std::io::stdin()
            File::open("/dev/stdin").unwrap_or_else(|_| {
                // Fallback for Windows or when /dev/stdin doesn't exist
                File::open("NUL").unwrap_or_else(|_| panic!("Cannot open stdin"))
            }),
        )))),
        mode: OpenMode::Read,
        path: "<stdin>".to_string(),
    }
}

/// Get a handle to standard output
pub fn stdout() -> Handle {
    Handle {
        inner: Arc::new(Mutex::new(HandleInner::Writer(BufWriter::new(
            OpenOptions::new()
                .write(true)
                .open("/dev/stdout")
                .unwrap_or_else(|_| {
                    OpenOptions::new()
                        .write(true)
                        .open("CON")
                        .unwrap_or_else(|_| panic!("Cannot open stdout"))
                }),
        )))),
        mode: OpenMode::Write,
        path: "<stdout>".to_string(),
    }
}

/// Get a handle to standard error
pub fn stderr() -> Handle {
    Handle {
        inner: Arc::new(Mutex::new(HandleInner::Writer(BufWriter::new(
            OpenOptions::new()
                .write(true)
                .open("/dev/stderr")
                .unwrap_or_else(|_| {
                    OpenOptions::new()
                        .write(true)
                        .open("CON")
                        .unwrap_or_else(|_| panic!("Cannot open stderr"))
                }),
        )))),
        mode: OpenMode::Write,
        path: "<stderr>".to_string(),
    }
}

// Convenience functions

/// Read an entire file as a string
///
/// # Example
///
/// ```no_run
/// use bhc_system::io::read_file;
///
/// let contents = read_file("config.txt").unwrap();
/// println!("Config: {}", contents);
/// ```
pub fn read_file<P: AsRef<Path>>(path: P) -> IoResult<String> {
    std::fs::read_to_string(path).map_err(Into::into)
}

/// Read an entire file as bytes
pub fn read_file_bytes<P: AsRef<Path>>(path: P) -> IoResult<Vec<u8>> {
    std::fs::read(path).map_err(Into::into)
}

/// Write a string to a file (creates or truncates)
///
/// # Example
///
/// ```no_run
/// use bhc_system::io::write_file;
///
/// write_file("output.txt", "Hello, World!").unwrap();
/// ```
pub fn write_file<P: AsRef<Path>>(path: P, contents: &str) -> IoResult<()> {
    std::fs::write(path, contents).map_err(Into::into)
}

/// Write bytes to a file (creates or truncates)
pub fn write_file_bytes<P: AsRef<Path>>(path: P, contents: &[u8]) -> IoResult<()> {
    std::fs::write(path, contents).map_err(Into::into)
}

/// Append a string to a file (creates if doesn't exist)
pub fn append_file<P: AsRef<Path>>(path: P, contents: &str) -> IoResult<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(contents.as_bytes())?;
    Ok(())
}

/// Append bytes to a file (creates if doesn't exist)
pub fn append_file_bytes<P: AsRef<Path>>(path: P, contents: &[u8]) -> IoResult<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(contents)?;
    Ok(())
}

/// Read lines from a file
pub fn read_lines<P: AsRef<Path>>(path: P) -> IoResult<Vec<String>> {
    let contents = read_file(path)?;
    Ok(contents.lines().map(String::from).collect())
}

/// Copy a file
pub fn copy_file<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> IoResult<u64> {
    std::fs::copy(from, to).map_err(Into::into)
}

// FFI exports for BHC runtime

/// Read file contents (FFI)
#[no_mangle]
pub extern "C" fn bhc_read_file(path: *const i8, out_len: *mut usize) -> *mut u8 {
    use std::ffi::CStr;

    if path.is_null() {
        return std::ptr::null_mut();
    }

    let path = unsafe { CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match read_file_bytes(path) {
        Ok(bytes) => {
            let len = bytes.len();
            let ptr = bytes.leak().as_mut_ptr();
            if !out_len.is_null() {
                unsafe { *out_len = len };
            }
            ptr
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Write file contents (FFI)
#[no_mangle]
pub extern "C" fn bhc_write_file(path: *const i8, data: *const u8, len: usize) -> i32 {
    use std::ffi::CStr;

    if path.is_null() || data.is_null() {
        return -1;
    }

    let path = unsafe { CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let data = unsafe { std::slice::from_raw_parts(data, len) };

    match write_file_bytes(path, data) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_write_and_read_file() {
        let path = "/tmp/bhc_io_test_1.txt";
        let content = "Hello, BHC!";

        write_file(path, content).unwrap();
        let read_content = read_file(path).unwrap();

        assert_eq!(read_content, content);
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_and_read_bytes() {
        let path = "/tmp/bhc_io_test_2.bin";
        let content = vec![0u8, 1, 2, 3, 4, 5, 255];

        write_file_bytes(path, &content).unwrap();
        let read_content = read_file_bytes(path).unwrap();

        assert_eq!(read_content, content);
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_append_file() {
        let path = "/tmp/bhc_io_test_3.txt";

        write_file(path, "Hello").unwrap();
        append_file(path, ", World!").unwrap();
        let content = read_file(path).unwrap();

        assert_eq!(content, "Hello, World!");
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_handle_read_write() {
        let path = "/tmp/bhc_io_test_4.txt";

        // Write using handle
        {
            let mut handle = Handle::open(path, OpenMode::Write).unwrap();
            handle.write_all(b"Test content").unwrap();
            handle.flush().unwrap();
        }

        // Read using handle
        {
            let mut handle = Handle::open(path, OpenMode::Read).unwrap();
            let content = handle.read_all().unwrap();
            assert_eq!(content, b"Test content");
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_handle_seek() {
        let path = "/tmp/bhc_io_test_5.txt";
        write_file(path, "0123456789").unwrap();

        let mut handle = Handle::open(path, OpenMode::Read).unwrap();

        // Seek to position 5
        handle.seek(SeekPosition::Start(5)).unwrap();
        let mut buf = [0u8; 5];
        handle.read(&mut buf).unwrap();
        assert_eq!(&buf, b"56789");

        // Seek back to start
        handle.seek(SeekPosition::Start(0)).unwrap();
        handle.read(&mut buf).unwrap();
        assert_eq!(&buf, b"01234");

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_read_lines() {
        let path = "/tmp/bhc_io_test_6.txt";
        write_file(path, "line1\nline2\nline3").unwrap();

        let lines = read_lines(path).unwrap();
        assert_eq!(lines, vec!["line1", "line2", "line3"]);

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_copy_file() {
        let src = "/tmp/bhc_io_test_7_src.txt";
        let dst = "/tmp/bhc_io_test_7_dst.txt";

        write_file(src, "copy me").unwrap();
        copy_file(src, dst).unwrap();

        let content = read_file(dst).unwrap();
        assert_eq!(content, "copy me");

        fs::remove_file(src).ok();
        fs::remove_file(dst).ok();
    }

    #[test]
    fn test_io_error_not_found() {
        let result = read_file("/nonexistent/path/file.txt");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, IoErrorKind::NotFound);
    }

    #[test]
    fn test_handle_position() {
        let path = "/tmp/bhc_io_test_8.txt";
        write_file(path, "0123456789").unwrap();

        let mut handle = Handle::open(path, OpenMode::Read).unwrap();
        assert_eq!(handle.position().unwrap(), 0);

        let mut buf = [0u8; 3];
        handle.read(&mut buf).unwrap();
        assert_eq!(handle.position().unwrap(), 3);

        handle.read(&mut buf).unwrap();
        assert_eq!(handle.position().unwrap(), 6);

        fs::remove_file(path).ok();
    }
}
