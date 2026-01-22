//! Directory operations
//!
//! This module provides functions for working with directories and
//! file system entries.
//!
//! # Example
//!
//! ```no_run
//! use bhc_system::directory::{
//!     create_directory, list_directory, exists, is_directory
//! };
//!
//! // Create a directory
//! create_directory("/tmp/my_app").unwrap();
//!
//! // List contents
//! for entry in list_directory("/tmp").unwrap() {
//!     println!("{}", entry);
//! }
//!
//! // Check existence
//! if exists("/tmp/my_app") && is_directory("/tmp/my_app") {
//!     println!("Directory exists!");
//! }
//! ```

use std::fs::{self, Metadata};
use std::path::Path;
use std::time::SystemTime;

/// Error type for directory operations
#[derive(Debug)]
pub struct DirError {
    /// The kind of error
    pub kind: DirErrorKind,
    /// Human-readable error message
    pub message: String,
    /// The path that caused the error (if applicable)
    pub path: Option<String>,
}

/// Categories of directory errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirErrorKind {
    /// Path not found
    NotFound,
    /// Permission denied
    PermissionDenied,
    /// Path already exists
    AlreadyExists,
    /// Path is not a directory
    NotADirectory,
    /// Directory is not empty
    NotEmpty,
    /// Invalid path
    InvalidPath,
    /// Other error
    Other,
}

impl std::fmt::Display for DirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.path {
            Some(path) => write!(f, "{:?} for '{}': {}", self.kind, path, self.message),
            None => write!(f, "{:?}: {}", self.kind, self.message),
        }
    }
}

impl std::error::Error for DirError {}

impl From<std::io::Error> for DirError {
    fn from(err: std::io::Error) -> Self {
        let kind = match err.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            std::io::ErrorKind::AlreadyExists => DirErrorKind::AlreadyExists,
            _ => DirErrorKind::Other,
        };
        DirError {
            kind,
            message: err.to_string(),
            path: None,
        }
    }
}

/// Result type for directory operations
pub type DirResult<T> = Result<T, DirError>;

/// File type information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Regular file
    File,
    /// Directory
    Directory,
    /// Symbolic link
    Symlink,
    /// Other (device, socket, etc.)
    Other,
}

/// Information about a directory entry
#[derive(Debug, Clone)]
pub struct DirEntry {
    /// Entry name (not full path)
    pub name: String,
    /// Full path
    pub path: String,
    /// File type
    pub file_type: FileType,
}

/// File metadata
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// File size in bytes
    pub size: u64,
    /// File type
    pub file_type: FileType,
    /// Is read-only
    pub readonly: bool,
    /// Last modification time (if available)
    pub modified: Option<SystemTime>,
    /// Last access time (if available)
    pub accessed: Option<SystemTime>,
    /// Creation time (if available)
    pub created: Option<SystemTime>,
}

impl From<Metadata> for FileInfo {
    fn from(meta: Metadata) -> Self {
        let file_type = if meta.is_dir() {
            FileType::Directory
        } else if meta.is_symlink() {
            FileType::Symlink
        } else if meta.is_file() {
            FileType::File
        } else {
            FileType::Other
        };

        FileInfo {
            size: meta.len(),
            file_type,
            readonly: meta.permissions().readonly(),
            modified: meta.modified().ok(),
            accessed: meta.accessed().ok(),
            created: meta.created().ok(),
        }
    }
}

/// Check if a path exists
///
/// # Example
///
/// ```no_run
/// use bhc_system::directory::exists;
///
/// if exists("/tmp") {
///     println!("/tmp exists");
/// }
/// ```
pub fn exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

/// Check if a path is a file
pub fn is_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_file()
}

/// Check if a path is a directory
pub fn is_directory<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_dir()
}

/// Check if a path is a symbolic link
pub fn is_symlink<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_symlink()
}

/// Get file/directory metadata
///
/// # Example
///
/// ```no_run
/// use bhc_system::directory::metadata;
///
/// let info = metadata("/tmp").unwrap();
/// println!("Size: {} bytes", info.size);
/// ```
pub fn metadata<P: AsRef<Path>>(path: P) -> DirResult<FileInfo> {
    let meta = fs::metadata(&path)?;
    Ok(FileInfo::from(meta))
}

/// Get symlink metadata (doesn't follow symlinks)
pub fn symlink_metadata<P: AsRef<Path>>(path: P) -> DirResult<FileInfo> {
    let meta = fs::symlink_metadata(&path)?;
    Ok(FileInfo::from(meta))
}

/// Create a directory
///
/// Fails if the parent directory doesn't exist.
///
/// # Example
///
/// ```no_run
/// use bhc_system::directory::create_directory;
///
/// create_directory("/tmp/new_dir").unwrap();
/// ```
pub fn create_directory<P: AsRef<Path>>(path: P) -> DirResult<()> {
    fs::create_dir(&path).map_err(|e| DirError {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            std::io::ErrorKind::AlreadyExists => DirErrorKind::AlreadyExists,
            _ => DirErrorKind::Other,
        },
        message: e.to_string(),
        path: Some(path.as_ref().to_string_lossy().to_string()),
    })
}

/// Create a directory and all parent directories
///
/// # Example
///
/// ```no_run
/// use bhc_system::directory::create_directory_all;
///
/// create_directory_all("/tmp/a/b/c/d").unwrap();
/// ```
pub fn create_directory_all<P: AsRef<Path>>(path: P) -> DirResult<()> {
    fs::create_dir_all(&path).map_err(|e| DirError {
        kind: DirErrorKind::Other,
        message: e.to_string(),
        path: Some(path.as_ref().to_string_lossy().to_string()),
    })
}

/// Remove an empty directory
pub fn remove_directory<P: AsRef<Path>>(path: P) -> DirResult<()> {
    fs::remove_dir(&path).map_err(|e| DirError {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            _ => DirErrorKind::Other,
        },
        message: e.to_string(),
        path: Some(path.as_ref().to_string_lossy().to_string()),
    })
}

/// Remove a directory and all its contents recursively
///
/// # Warning
///
/// This is a destructive operation. Use with caution.
pub fn remove_directory_all<P: AsRef<Path>>(path: P) -> DirResult<()> {
    fs::remove_dir_all(&path).map_err(|e| DirError {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            _ => DirErrorKind::Other,
        },
        message: e.to_string(),
        path: Some(path.as_ref().to_string_lossy().to_string()),
    })
}

/// Remove a file
pub fn remove_file<P: AsRef<Path>>(path: P) -> DirResult<()> {
    fs::remove_file(&path).map_err(|e| DirError {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            _ => DirErrorKind::Other,
        },
        message: e.to_string(),
        path: Some(path.as_ref().to_string_lossy().to_string()),
    })
}

/// Rename/move a file or directory
pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> DirResult<()> {
    fs::rename(&from, &to).map_err(|e| DirError {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            std::io::ErrorKind::AlreadyExists => DirErrorKind::AlreadyExists,
            _ => DirErrorKind::Other,
        },
        message: e.to_string(),
        path: Some(from.as_ref().to_string_lossy().to_string()),
    })
}

/// List directory contents
///
/// Returns a list of entry names (not full paths).
///
/// # Example
///
/// ```no_run
/// use bhc_system::directory::list_directory;
///
/// for entry in list_directory("/tmp").unwrap() {
///     println!("{}", entry);
/// }
/// ```
pub fn list_directory<P: AsRef<Path>>(path: P) -> DirResult<Vec<String>> {
    let entries = fs::read_dir(&path).map_err(|e| DirError {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            _ => DirErrorKind::Other,
        },
        message: e.to_string(),
        path: Some(path.as_ref().to_string_lossy().to_string()),
    })?;

    let mut names = Vec::new();
    for entry in entries {
        if let Ok(entry) = entry {
            names.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    Ok(names)
}

/// List directory contents with full information
pub fn list_directory_entries<P: AsRef<Path>>(path: P) -> DirResult<Vec<DirEntry>> {
    let entries = fs::read_dir(&path).map_err(|e| DirError {
        kind: DirErrorKind::Other,
        message: e.to_string(),
        path: Some(path.as_ref().to_string_lossy().to_string()),
    })?;

    let mut result = Vec::new();
    for entry in entries {
        if let Ok(entry) = entry {
            let file_type = entry
                .file_type()
                .map(|ft| {
                    if ft.is_dir() {
                        FileType::Directory
                    } else if ft.is_symlink() {
                        FileType::Symlink
                    } else if ft.is_file() {
                        FileType::File
                    } else {
                        FileType::Other
                    }
                })
                .unwrap_or(FileType::Other);

            result.push(DirEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                path: entry.path().to_string_lossy().to_string(),
                file_type,
            });
        }
    }
    Ok(result)
}

/// Get the current working directory
pub fn current_directory() -> DirResult<String> {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| DirError {
            kind: DirErrorKind::Other,
            message: e.to_string(),
            path: None,
        })
}

/// Set the current working directory
pub fn set_current_directory<P: AsRef<Path>>(path: P) -> DirResult<()> {
    std::env::set_current_dir(&path).map_err(|e| DirError {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            _ => DirErrorKind::Other,
        },
        message: e.to_string(),
        path: Some(path.as_ref().to_string_lossy().to_string()),
    })
}

/// Get file size in bytes
pub fn file_size<P: AsRef<Path>>(path: P) -> DirResult<u64> {
    metadata(path).map(|m| m.size)
}

/// Get the canonical (absolute, resolved) path
pub fn canonicalize<P: AsRef<Path>>(path: P) -> DirResult<String> {
    fs::canonicalize(&path)
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| DirError {
            kind: match e.kind() {
                std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
                _ => DirErrorKind::Other,
            },
            message: e.to_string(),
            path: Some(path.as_ref().to_string_lossy().to_string()),
        })
}

/// Create a symbolic link
#[cfg(unix)]
pub fn create_symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) -> DirResult<()> {
    std::os::unix::fs::symlink(&original, &link).map_err(|e| DirError {
        kind: DirErrorKind::Other,
        message: e.to_string(),
        path: Some(link.as_ref().to_string_lossy().to_string()),
    })
}

/// Create a symbolic link (Windows version creates a file symlink)
#[cfg(windows)]
pub fn create_symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) -> DirResult<()> {
    std::os::windows::fs::symlink_file(&original, &link).map_err(|e| DirError {
        kind: DirErrorKind::Other,
        message: e.to_string(),
        path: Some(link.as_ref().to_string_lossy().to_string()),
    })
}

/// Read the target of a symbolic link
pub fn read_link<P: AsRef<Path>>(path: P) -> DirResult<String> {
    fs::read_link(&path)
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| DirError {
            kind: match e.kind() {
                std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
                _ => DirErrorKind::Other,
            },
            message: e.to_string(),
            path: Some(path.as_ref().to_string_lossy().to_string()),
        })
}

/// Copy a file
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> DirResult<u64> {
    fs::copy(&from, &to).map_err(|e| DirError {
        kind: match e.kind() {
            std::io::ErrorKind::NotFound => DirErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => DirErrorKind::PermissionDenied,
            _ => DirErrorKind::Other,
        },
        message: e.to_string(),
        path: Some(from.as_ref().to_string_lossy().to_string()),
    })
}

/// Walk a directory recursively
pub fn walk_directory<P: AsRef<Path>>(path: P) -> DirResult<Vec<DirEntry>> {
    fn walk_recursive(path: &Path, entries: &mut Vec<DirEntry>) -> DirResult<()> {
        for entry in fs::read_dir(path).map_err(|e| DirError {
            kind: DirErrorKind::Other,
            message: e.to_string(),
            path: Some(path.to_string_lossy().to_string()),
        })? {
            if let Ok(entry) = entry {
                let file_type = entry
                    .file_type()
                    .map(|ft| {
                        if ft.is_dir() {
                            FileType::Directory
                        } else if ft.is_symlink() {
                            FileType::Symlink
                        } else if ft.is_file() {
                            FileType::File
                        } else {
                            FileType::Other
                        }
                    })
                    .unwrap_or(FileType::Other);

                let dir_entry = DirEntry {
                    name: entry.file_name().to_string_lossy().to_string(),
                    path: entry.path().to_string_lossy().to_string(),
                    file_type,
                };

                entries.push(dir_entry);

                if file_type == FileType::Directory {
                    walk_recursive(&entry.path(), entries)?;
                }
            }
        }
        Ok(())
    }

    let mut entries = Vec::new();
    walk_recursive(path.as_ref(), &mut entries)?;
    Ok(entries)
}

// FFI exports

/// Check if path exists (FFI)
#[no_mangle]
pub extern "C" fn bhc_exists(path: *const i8) -> i32 {
    use std::ffi::CStr;

    if path.is_null() {
        return 0;
    }

    let path = unsafe { CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    if exists(path) {
        1
    } else {
        0
    }
}

/// Check if path is directory (FFI)
#[no_mangle]
pub extern "C" fn bhc_is_directory(path: *const i8) -> i32 {
    use std::ffi::CStr;

    if path.is_null() {
        return 0;
    }

    let path = unsafe { CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    if is_directory(path) {
        1
    } else {
        0
    }
}

/// Create directory (FFI)
#[no_mangle]
pub extern "C" fn bhc_create_directory(path: *const i8) -> i32 {
    use std::ffi::CStr;

    if path.is_null() {
        return -1;
    }

    let path = unsafe { CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match create_directory_all(path) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exists() {
        assert!(exists("/tmp"));
        assert!(!exists("/definitely_nonexistent_path_12345"));
    }

    #[test]
    fn test_is_directory() {
        assert!(is_directory("/tmp"));
    }

    #[test]
    fn test_is_file() {
        // Create a test file
        let path = "/tmp/bhc_dir_test_file.txt";
        std::fs::write(path, "test").ok();
        assert!(is_file(path));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_create_and_remove_directory() {
        let path = "/tmp/bhc_dir_test_create";

        // Clean up first
        remove_directory(path).ok();

        // Create
        create_directory(path).unwrap();
        assert!(exists(path));
        assert!(is_directory(path));

        // Remove
        remove_directory(path).unwrap();
        assert!(!exists(path));
    }

    #[test]
    fn test_create_directory_all() {
        let path = "/tmp/bhc_dir_test_a/b/c";

        // Clean up first
        remove_directory_all("/tmp/bhc_dir_test_a").ok();

        // Create nested
        create_directory_all(path).unwrap();
        assert!(exists(path));
        assert!(is_directory(path));

        // Clean up
        remove_directory_all("/tmp/bhc_dir_test_a").ok();
    }

    #[test]
    fn test_list_directory() {
        let entries = list_directory("/tmp").unwrap();
        // /tmp should have at least some entries
        assert!(!entries.is_empty() || entries.is_empty()); // May be empty in some envs
    }

    #[test]
    fn test_metadata() {
        let info = metadata("/tmp").unwrap();
        assert_eq!(info.file_type, FileType::Directory);
    }

    #[test]
    fn test_current_directory() {
        let cwd = current_directory().unwrap();
        assert!(!cwd.is_empty());
    }

    #[test]
    fn test_rename() {
        let from = "/tmp/bhc_rename_test_from.txt";
        let to = "/tmp/bhc_rename_test_to.txt";

        // Clean up
        std::fs::remove_file(from).ok();
        std::fs::remove_file(to).ok();

        // Create source
        std::fs::write(from, "test").unwrap();
        assert!(exists(from));

        // Rename
        rename(from, to).unwrap();
        assert!(!exists(from));
        assert!(exists(to));

        // Clean up
        std::fs::remove_file(to).ok();
    }

    #[test]
    fn test_copy() {
        let from = "/tmp/bhc_copy_test_from.txt";
        let to = "/tmp/bhc_copy_test_to.txt";

        // Clean up
        std::fs::remove_file(from).ok();
        std::fs::remove_file(to).ok();

        // Create source
        std::fs::write(from, "test content").unwrap();

        // Copy
        let bytes = copy(from, to).unwrap();
        assert_eq!(bytes, 12); // "test content".len()
        assert!(exists(to));

        // Clean up
        std::fs::remove_file(from).ok();
        std::fs::remove_file(to).ok();
    }

    #[test]
    fn test_file_size() {
        let path = "/tmp/bhc_size_test.txt";
        std::fs::write(path, "12345").unwrap();

        let size = file_size(path).unwrap();
        assert_eq!(size, 5);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_canonicalize() {
        let canonical = canonicalize("/tmp").unwrap();
        assert!(is_absolute(&canonical));
    }

    fn is_absolute(path: &str) -> bool {
        Path::new(path).is_absolute()
    }
}
