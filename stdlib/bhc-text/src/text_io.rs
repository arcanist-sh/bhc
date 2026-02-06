//! Data.Text.IO FFI primitives for BHC.
//!
//! Provides file and handle I/O operations that work directly with BHC's
//! Text representation, avoiding the pack/unpack overhead of String-based I/O.
//!
//! File path functions (readFile, writeFile, appendFile) take C-string paths
//! since the codegen already converts `[Char]` to C-strings.
//!
//! Handle functions use the same sentinel-pointer convention as bhc-rts:
//! 1=stdin, 2=stdout, 3=stderr, otherwise heap-allocated BhcHandle.

use std::ffi::CStr;
use std::fs;
use std::io::{BufRead, Read, Write};
use std::os::raw::c_char;

use crate::text::{alloc_text_from_bytes, text_bytes};

// ============================================================
// Handle sentinel constants (must match bhc-rts/src/ffi.rs)
// ============================================================

const HANDLE_STDIN: usize = 1;
const HANDLE_STDOUT: usize = 2;
const HANDLE_STDERR: usize = 3;

/// Mirror of BhcHandle from bhc-rts. Must have identical layout.
#[repr(C)]
struct BhcHandle {
    file: Option<std::fs::File>,
    readable: bool,
    writable: bool,
    closed: bool,
}

fn is_sentinel(handle: *mut u8) -> bool {
    let h = handle as usize;
    h == HANDLE_STDIN || h == HANDLE_STDOUT || h == HANDLE_STDERR
}

// ============================================================
// File path operations
// ============================================================

/// Read a file's contents as Text.
///
/// `path` is a null-terminated C string (codegen converts `[Char]` path).
/// Returns a BhcText pointer containing the file's UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn bhc_text_read_file(path: *const c_char) -> *mut u8 {
    if path.is_null() {
        return alloc_text_from_bytes(&[]);
    }
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return alloc_text_from_bytes(&[]),
    };
    match fs::read(path_str) {
        Ok(bytes) => alloc_text_from_bytes(&bytes),
        Err(_) => alloc_text_from_bytes(&[]),
    }
}

/// Write Text to a file (create/truncate).
///
/// `path` is a null-terminated C string. `text` is a BhcText pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_text_write_file(path: *const c_char, text: *const u8) {
    if path.is_null() {
        return;
    }
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return,
    };
    let bytes = if text.is_null() {
        &[] as &[u8]
    } else {
        text_bytes(text)
    };
    let _ = fs::write(path_str, bytes);
}

/// Append Text to a file.
///
/// `path` is a null-terminated C string. `text` is a BhcText pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_text_append_file(path: *const c_char, text: *const u8) {
    if path.is_null() {
        return;
    }
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return,
    };
    let bytes = if text.is_null() {
        &[] as &[u8]
    } else {
        text_bytes(text)
    };
    let mut file = match fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path_str)
    {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = file.write_all(bytes);
}

// ============================================================
// Handle operations (direct I/O, matching RTS sentinel pattern)
// ============================================================

/// Read all remaining contents from a handle as Text.
///
/// `handle` is a BHC handle pointer (sentinel or BhcHandle).
#[no_mangle]
pub unsafe extern "C" fn bhc_text_h_get_contents(handle: *mut u8) -> *mut u8 {
    let h = handle as usize;
    if h == HANDLE_STDIN {
        let mut buf = Vec::new();
        let _ = std::io::stdin().lock().read_to_end(&mut buf);
        return alloc_text_from_bytes(&buf);
    }
    if !handle.is_null() && !is_sentinel(handle) {
        let bh = &mut *(handle as *mut BhcHandle);
        if let Some(ref mut f) = bh.file {
            let mut buf = Vec::new();
            let _ = f.read_to_end(&mut buf);
            return alloc_text_from_bytes(&buf);
        }
    }
    alloc_text_from_bytes(&[])
}

/// Read a line from a handle as Text (without trailing newline).
///
/// `handle` is a BHC handle pointer (sentinel or BhcHandle).
#[no_mangle]
pub unsafe extern "C" fn bhc_text_h_get_line(handle: *mut u8) -> *mut u8 {
    let h = handle as usize;
    if h == HANDLE_STDIN {
        let mut line = String::new();
        let _ = std::io::stdin().lock().read_line(&mut line);
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        return alloc_text_from_bytes(line.as_bytes());
    }
    if !handle.is_null() && !is_sentinel(handle) {
        let bh = &mut *(handle as *mut BhcHandle);
        if let Some(ref mut f) = bh.file {
            let mut reader = std::io::BufReader::new(f);
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            return alloc_text_from_bytes(line.as_bytes());
        }
    }
    alloc_text_from_bytes(&[])
}

/// Write Text bytes to a handle.
///
/// `handle` is a BHC handle pointer. `text` is a BhcText pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_text_h_put_str(handle: *mut u8, text: *const u8) {
    let bytes = if text.is_null() {
        &[] as &[u8]
    } else {
        text_bytes(text)
    };
    if bytes.is_empty() {
        return;
    }
    let h = handle as usize;
    if h == HANDLE_STDOUT {
        let _ = std::io::stdout().write_all(bytes);
    } else if h == HANDLE_STDERR {
        let _ = std::io::stderr().write_all(bytes);
    } else if !handle.is_null() && !is_sentinel(handle) {
        let bh = &mut *(handle as *mut BhcHandle);
        if let Some(ref mut f) = bh.file {
            let _ = f.write_all(bytes);
        }
    }
}

/// Write Text bytes + newline to a handle.
///
/// `handle` is a BHC handle pointer. `text` is a BhcText pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_text_h_put_str_ln(handle: *mut u8, text: *const u8) {
    let bytes = if text.is_null() {
        &[] as &[u8]
    } else {
        text_bytes(text)
    };
    let h = handle as usize;
    if h == HANDLE_STDOUT {
        let _ = std::io::stdout().write_all(bytes);
        let _ = std::io::stdout().write_all(b"\n");
    } else if h == HANDLE_STDERR {
        let _ = std::io::stderr().write_all(bytes);
        let _ = std::io::stderr().write_all(b"\n");
    } else if !handle.is_null() && !is_sentinel(handle) {
        let bh = &mut *(handle as *mut BhcHandle);
        if let Some(ref mut f) = bh.file {
            let _ = f.write_all(bytes);
            let _ = f.write_all(b"\n");
        }
    }
}
