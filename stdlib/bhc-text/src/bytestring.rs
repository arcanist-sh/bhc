//! ByteString FFI primitives for BHC Data.ByteString.
//!
//! These functions implement the `extern "C"` FFI interface for strict
//! bytestrings. ByteString uses **identical memory layout** to Text:
//!
//! # Representation
//!
//! ```text
//! [u64 data_ptr][u64 offset][u64 byte_len][...bytes...]
//! ```
//!
//! The header is 24 bytes. `data_ptr` points to the start of the
//! byte buffer (which may be shared across slices). This enables
//! zero-copy slicing and allows Text.Encoding to share the byte
//! buffer directly between Text and ByteString.

use std::alloc::{self, Layout};
use std::ptr;

/// Header: data_ptr (8) + offset (8) + byte_len (8) = 24 bytes.
const HEADER_SIZE: usize = 24;

// ============================================================
// Internal helpers
// ============================================================

/// Read the data pointer from a ByteString header.
unsafe fn bs_data_ptr(bs: *const u8) -> *const u8 {
    *(bs as *const *const u8)
}

/// Read the byte offset from a ByteString header.
unsafe fn bs_offset(bs: *const u8) -> usize {
    *((bs as *const u64).add(1)) as usize
}

/// Read the byte length from a ByteString header.
unsafe fn bs_byte_len(bs: *const u8) -> usize {
    *((bs as *const u64).add(2)) as usize
}

/// Get a slice view of the ByteString's active bytes.
unsafe fn bs_bytes(bs: *const u8) -> &'static [u8] {
    let data = bs_data_ptr(bs);
    let off = bs_offset(bs);
    let len = bs_byte_len(bs);
    std::slice::from_raw_parts(data.add(off), len)
}

/// Allocate a new self-contained ByteString from a byte slice.
fn alloc_bs_from_bytes(bytes: &[u8]) -> *mut u8 {
    let total = HEADER_SIZE + bytes.len();
    let layout = Layout::from_size_align(total, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            return ptr::null_mut();
        }
        // data_ptr = ptr + HEADER_SIZE
        let data_start = ptr.add(HEADER_SIZE);
        (ptr as *mut *mut u8).write(data_start);
        // offset = 0
        (ptr as *mut u64).add(1).write(0);
        // byte_len = bytes.len()
        (ptr as *mut u64).add(2).write(bytes.len() as u64);
        // Copy data
        if !bytes.is_empty() {
            ptr::copy_nonoverlapping(bytes.as_ptr(), data_start, bytes.len());
        }
        ptr
    }
}

/// Allocate a ByteString that is a slice (view) of another's data.
fn alloc_bs_slice(source_data_ptr: *const u8, offset: usize, byte_len: usize) -> *mut u8 {
    let layout = Layout::from_size_align(HEADER_SIZE, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            return ptr::null_mut();
        }
        (ptr as *mut *const u8).write(source_data_ptr);
        (ptr as *mut u64).add(1).write(offset as u64);
        (ptr as *mut u64).add(2).write(byte_len as u64);
        ptr
    }
}

// ============================================================
// Creation
// ============================================================

/// Return an empty ByteString.
#[no_mangle]
pub extern "C" fn bhc_bs_empty() -> *mut u8 {
    alloc_bs_from_bytes(&[])
}

/// Create a ByteString containing a single byte.
///
/// `byte` is passed as an i64 representing a Word8.
#[no_mangle]
pub extern "C" fn bhc_bs_singleton(byte: i64) -> *mut u8 {
    alloc_bs_from_bytes(&[byte as u8])
}

/// Pack a BHC cons-list of Word8 (as Int) into a ByteString.
///
/// The list uses the standard BHC ADT layout:
/// - Nil: tag == 0
/// - Cons head tail: tag == 1, fields[0] = head (Int), fields[1] = tail
#[no_mangle]
pub extern "C" fn bhc_bs_pack(list_ptr: *const u8) -> *mut u8 {
    if list_ptr.is_null() {
        return bhc_bs_empty();
    }
    let mut result = Vec::new();
    let mut current = list_ptr;
    unsafe {
        loop {
            let tag = *(current as *const i64);
            if tag == 0 {
                break;
            }
            // Cons: tag == 1, fields at offset 8
            let fields_base = (current as *const *const u8).add(1);
            let head_raw = *fields_base;
            let byte_val = head_raw as i64 as u8;
            result.push(byte_val);
            current = *fields_base.add(1);
            if current.is_null() {
                break;
            }
        }
    }
    alloc_bs_from_bytes(&result)
}

// ============================================================
// Basic interface
// ============================================================

/// Return the byte at a given index (for unpack iteration).
///
/// Returns the byte value as i64.
#[no_mangle]
pub extern "C" fn bhc_bs_unpack(bs: *const u8, index: i64) -> i64 {
    if bs.is_null() || index < 0 {
        return 0;
    }
    unsafe {
        let bytes = bs_bytes(bs);
        let idx = index as usize;
        if idx < bytes.len() {
            bytes[idx] as i64
        } else {
            0
        }
    }
}

/// Return the number of bytes in a ByteString (O(1)).
#[no_mangle]
pub extern "C" fn bhc_bs_length(bs: *const u8) -> i64 {
    if bs.is_null() {
        return 0;
    }
    unsafe { bs_byte_len(bs) as i64 }
}

/// Test whether a ByteString is empty.
///
/// Returns 1 for true, 0 for false.
#[no_mangle]
pub extern "C" fn bhc_bs_null(bs: *const u8) -> i64 {
    if bs.is_null() {
        return 1;
    }
    unsafe {
        if bs_byte_len(bs) == 0 {
            1
        } else {
            0
        }
    }
}

/// Extract the first byte.
#[no_mangle]
pub extern "C" fn bhc_bs_head(bs: *const u8) -> i64 {
    if bs.is_null() {
        return 0;
    }
    unsafe {
        let bytes = bs_bytes(bs);
        if bytes.is_empty() {
            0
        } else {
            bytes[0] as i64
        }
    }
}

/// Extract the last byte.
#[no_mangle]
pub extern "C" fn bhc_bs_last(bs: *const u8) -> i64 {
    if bs.is_null() {
        return 0;
    }
    unsafe {
        let bytes = bs_bytes(bs);
        if bytes.is_empty() {
            0
        } else {
            bytes[bytes.len() - 1] as i64
        }
    }
}

/// Return all bytes after the first (zero-copy slice).
#[no_mangle]
pub extern "C" fn bhc_bs_tail(bs: *const u8) -> *mut u8 {
    if bs.is_null() {
        return bhc_bs_empty();
    }
    unsafe {
        let len = bs_byte_len(bs);
        if len == 0 {
            return bhc_bs_empty();
        }
        let data = bs_data_ptr(bs);
        let off = bs_offset(bs) + 1;
        alloc_bs_slice(data, off, len - 1)
    }
}

/// Return all bytes except the last (zero-copy slice).
#[no_mangle]
pub extern "C" fn bhc_bs_init(bs: *const u8) -> *mut u8 {
    if bs.is_null() {
        return bhc_bs_empty();
    }
    unsafe {
        let len = bs_byte_len(bs);
        if len == 0 {
            return bhc_bs_empty();
        }
        let data = bs_data_ptr(bs);
        let off = bs_offset(bs);
        alloc_bs_slice(data, off, len - 1)
    }
}

// ============================================================
// Concatenation
// ============================================================

/// Append two ByteStrings.
#[no_mangle]
pub extern "C" fn bhc_bs_append(a: *const u8, b: *const u8) -> *mut u8 {
    let a_bytes = if a.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(a) }
    };
    let b_bytes = if b.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(b) }
    };

    if a_bytes.is_empty() && !b.is_null() {
        return unsafe { alloc_bs_slice(bs_data_ptr(b), bs_offset(b), bs_byte_len(b)) };
    }
    if b_bytes.is_empty() && !a.is_null() {
        return unsafe { alloc_bs_slice(bs_data_ptr(a), bs_offset(a), bs_byte_len(a)) };
    }

    let mut combined = Vec::with_capacity(a_bytes.len() + b_bytes.len());
    combined.extend_from_slice(a_bytes);
    combined.extend_from_slice(b_bytes);
    alloc_bs_from_bytes(&combined)
}

/// Prepend a byte to a ByteString.
#[no_mangle]
pub extern "C" fn bhc_bs_cons(byte: i64, bs: *const u8) -> *mut u8 {
    let tail_bytes = if bs.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(bs) }
    };
    let mut result = Vec::with_capacity(1 + tail_bytes.len());
    result.push(byte as u8);
    result.extend_from_slice(tail_bytes);
    alloc_bs_from_bytes(&result)
}

/// Append a byte to a ByteString.
#[no_mangle]
pub extern "C" fn bhc_bs_snoc(bs: *const u8, byte: i64) -> *mut u8 {
    let init_bytes = if bs.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(bs) }
    };
    let mut result = Vec::with_capacity(init_bytes.len() + 1);
    result.extend_from_slice(init_bytes);
    result.push(byte as u8);
    alloc_bs_from_bytes(&result)
}

// ============================================================
// Substrings
// ============================================================

/// Take the first `n` bytes (zero-copy slice).
#[no_mangle]
pub extern "C" fn bhc_bs_take(n: i64, bs: *const u8) -> *mut u8 {
    if bs.is_null() || n <= 0 {
        return bhc_bs_empty();
    }
    unsafe {
        let len = bs_byte_len(bs);
        let take_n = (n as usize).min(len);
        let data = bs_data_ptr(bs);
        let off = bs_offset(bs);
        alloc_bs_slice(data, off, take_n)
    }
}

/// Drop the first `n` bytes (zero-copy slice).
#[no_mangle]
pub extern "C" fn bhc_bs_drop(n: i64, bs: *const u8) -> *mut u8 {
    if bs.is_null() {
        return bhc_bs_empty();
    }
    if n <= 0 {
        return unsafe { alloc_bs_slice(bs_data_ptr(bs), bs_offset(bs), bs_byte_len(bs)) };
    }
    unsafe {
        let len = bs_byte_len(bs);
        let drop_n = (n as usize).min(len);
        let data = bs_data_ptr(bs);
        let off = bs_offset(bs) + drop_n;
        alloc_bs_slice(data, off, len - drop_n)
    }
}

// ============================================================
// Transformations
// ============================================================

/// Reverse a ByteString.
#[no_mangle]
pub extern "C" fn bhc_bs_reverse(bs: *const u8) -> *mut u8 {
    if bs.is_null() {
        return bhc_bs_empty();
    }
    unsafe {
        let bytes = bs_bytes(bs);
        let mut reversed: Vec<u8> = bytes.to_vec();
        reversed.reverse();
        alloc_bs_from_bytes(&reversed)
    }
}

// ============================================================
// Searching
// ============================================================

/// Test whether a byte occurs in the ByteString.
///
/// Returns 1 for true, 0 for false.
#[no_mangle]
pub extern "C" fn bhc_bs_elem(byte: i64, bs: *const u8) -> i64 {
    if bs.is_null() {
        return 0;
    }
    unsafe {
        let bytes = bs_bytes(bs);
        let needle = byte as u8;
        if bytes.contains(&needle) {
            1
        } else {
            0
        }
    }
}

/// Index into a ByteString, returning the byte at position `i`.
#[no_mangle]
pub extern "C" fn bhc_bs_index(bs: *const u8, i: i64) -> i64 {
    if bs.is_null() || i < 0 {
        return 0;
    }
    unsafe {
        let bytes = bs_bytes(bs);
        let idx = i as usize;
        if idx < bytes.len() {
            bytes[idx] as i64
        } else {
            0
        }
    }
}

// ============================================================
// Comparison
// ============================================================

/// Compare two ByteStrings for equality.
///
/// Returns 1 if equal, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_bs_eq(a: *const u8, b: *const u8) -> i64 {
    if a.is_null() && b.is_null() {
        return 1;
    }
    if a.is_null() || b.is_null() {
        return 0;
    }
    unsafe {
        if bs_bytes(a) == bs_bytes(b) {
            1
        } else {
            0
        }
    }
}

/// Lexicographic comparison of two ByteStrings.
///
/// Returns 0 (LT), 1 (EQ), or 2 (GT), matching Haskell `Ordering` tags.
#[no_mangle]
pub extern "C" fn bhc_bs_compare(a: *const u8, b: *const u8) -> i64 {
    use std::cmp::Ordering;
    let a_bytes = if a.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(a) }
    };
    let b_bytes = if b.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(b) }
    };
    match a_bytes.cmp(b_bytes) {
        Ordering::Less => 0,
        Ordering::Equal => 1,
        Ordering::Greater => 2,
    }
}

// ============================================================
// Predicates
// ============================================================

/// Test whether `prefix` is a prefix of `bs`.
///
/// Returns 1 for true, 0 for false.
#[no_mangle]
pub extern "C" fn bhc_bs_is_prefix_of(prefix: *const u8, bs: *const u8) -> i64 {
    let p_bytes = if prefix.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(prefix) }
    };
    let b_bytes = if bs.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(bs) }
    };
    if b_bytes.starts_with(p_bytes) {
        1
    } else {
        0
    }
}

/// Test whether `suffix` is a suffix of `bs`.
///
/// Returns 1 for true, 0 for false.
#[no_mangle]
pub extern "C" fn bhc_bs_is_suffix_of(suffix: *const u8, bs: *const u8) -> i64 {
    let s_bytes = if suffix.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(suffix) }
    };
    let b_bytes = if bs.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(bs) }
    };
    if b_bytes.ends_with(s_bytes) {
        1
    } else {
        0
    }
}

// ============================================================
// IO operations
// ============================================================

/// Read a file into a ByteString.
///
/// Takes a BHC cons-list of Char (the file path) and returns a ByteString.
#[no_mangle]
pub extern "C" fn bhc_bs_read_file(path_ptr: *const u8) -> *mut u8 {
    if path_ptr.is_null() {
        return bhc_bs_empty();
    }
    // Walk the cons-list to extract the path string
    let mut path_str = String::new();
    let mut current = path_ptr;
    unsafe {
        loop {
            let tag = *(current as *const i64);
            if tag == 0 {
                break;
            }
            let fields_base = (current as *const *const u8).add(1);
            let head_raw = *fields_base;
            let codepoint = head_raw as i64;
            if let Some(c) = char::from_u32(codepoint as u32) {
                path_str.push(c);
            }
            current = *fields_base.add(1);
            if current.is_null() {
                break;
            }
        }
    }
    match std::fs::read(&path_str) {
        Ok(bytes) => alloc_bs_from_bytes(&bytes),
        Err(_) => bhc_bs_empty(),
    }
}

/// Write a ByteString to a file.
///
/// Takes a BHC cons-list of Char (the file path) and a ByteString.
#[no_mangle]
pub extern "C" fn bhc_bs_write_file(path_ptr: *const u8, bs: *const u8) {
    if path_ptr.is_null() {
        return;
    }
    let mut path_str = String::new();
    let mut current = path_ptr;
    unsafe {
        loop {
            let tag = *(current as *const i64);
            if tag == 0 {
                break;
            }
            let fields_base = (current as *const *const u8).add(1);
            let head_raw = *fields_base;
            let codepoint = head_raw as i64;
            if let Some(c) = char::from_u32(codepoint as u32) {
                path_str.push(c);
            }
            current = *fields_base.add(1);
            if current.is_null() {
                break;
            }
        }
    }
    let bytes = if bs.is_null() {
        &[]
    } else {
        unsafe { bs_bytes(bs) }
    };
    let _ = std::fs::write(&path_str, bytes);
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bs(bytes: &[u8]) -> *mut u8 {
        alloc_bs_from_bytes(bytes)
    }

    #[test]
    fn test_empty() {
        let bs = bhc_bs_empty();
        assert_eq!(bhc_bs_null(bs), 1);
        assert_eq!(bhc_bs_length(bs), 0);
    }

    #[test]
    fn test_singleton() {
        let bs = bhc_bs_singleton(42);
        assert_eq!(bhc_bs_null(bs), 0);
        assert_eq!(bhc_bs_length(bs), 1);
        assert_eq!(bhc_bs_head(bs), 42);
    }

    #[test]
    fn test_length() {
        let bs = make_bs(b"Hello");
        assert_eq!(bhc_bs_length(bs), 5);
    }

    #[test]
    fn test_head_last() {
        let bs = make_bs(b"Hello");
        assert_eq!(bhc_bs_head(bs), b'H' as i64);
        assert_eq!(bhc_bs_last(bs), b'o' as i64);
    }

    #[test]
    fn test_tail() {
        let bs = make_bs(b"Hello");
        let tail = bhc_bs_tail(bs);
        assert_eq!(bhc_bs_length(tail), 4);
        assert_eq!(bhc_bs_head(tail), b'e' as i64);
    }

    #[test]
    fn test_init() {
        let bs = make_bs(b"Hello");
        let init = bhc_bs_init(bs);
        assert_eq!(bhc_bs_length(init), 4);
        assert_eq!(bhc_bs_last(init), b'l' as i64);
    }

    #[test]
    fn test_append() {
        let a = make_bs(b"Hello");
        let b = make_bs(b" World");
        let c = bhc_bs_append(a, b);
        assert_eq!(bhc_bs_length(c), 11);
        assert_eq!(bhc_bs_head(c), b'H' as i64);
        assert_eq!(bhc_bs_last(c), b'd' as i64);
    }

    #[test]
    fn test_cons_snoc() {
        let bs = make_bs(b"ello");
        let consed = bhc_bs_cons(b'H' as i64, bs);
        assert_eq!(bhc_bs_length(consed), 5);
        assert_eq!(bhc_bs_head(consed), b'H' as i64);

        let snoced = bhc_bs_snoc(consed, b'!' as i64);
        assert_eq!(bhc_bs_length(snoced), 6);
        assert_eq!(bhc_bs_last(snoced), b'!' as i64);
    }

    #[test]
    fn test_take_drop() {
        let bs = make_bs(b"Hello World");
        let taken = bhc_bs_take(5, bs);
        assert_eq!(bhc_bs_length(taken), 5);
        assert_eq!(bhc_bs_head(taken), b'H' as i64);

        let dropped = bhc_bs_drop(6, bs);
        assert_eq!(bhc_bs_length(dropped), 5);
        assert_eq!(bhc_bs_head(dropped), b'W' as i64);
    }

    #[test]
    fn test_reverse() {
        let bs = make_bs(b"Hello");
        let rev = bhc_bs_reverse(bs);
        assert_eq!(bhc_bs_head(rev), b'o' as i64);
        assert_eq!(bhc_bs_last(rev), b'H' as i64);
    }

    #[test]
    fn test_elem() {
        let bs = make_bs(b"Hello");
        assert_eq!(bhc_bs_elem(b'H' as i64, bs), 1);
        assert_eq!(bhc_bs_elem(b'z' as i64, bs), 0);
    }

    #[test]
    fn test_index() {
        let bs = make_bs(b"Hello");
        assert_eq!(bhc_bs_index(bs, 0), b'H' as i64);
        assert_eq!(bhc_bs_index(bs, 4), b'o' as i64);
    }

    #[test]
    fn test_eq() {
        let a = make_bs(b"hello");
        let b = make_bs(b"hello");
        let c = make_bs(b"world");
        assert_eq!(bhc_bs_eq(a, b), 1);
        assert_eq!(bhc_bs_eq(a, c), 0);
    }

    #[test]
    fn test_compare() {
        let a = make_bs(b"abc");
        let b = make_bs(b"abd");
        let c = make_bs(b"abc");
        assert_eq!(bhc_bs_compare(a, b), 0); // LT
        assert_eq!(bhc_bs_compare(a, c), 1); // EQ
        assert_eq!(bhc_bs_compare(b, a), 2); // GT
    }

    #[test]
    fn test_is_prefix_of() {
        let prefix = make_bs(b"Hello");
        let text = make_bs(b"Hello World");
        let other = make_bs(b"World");
        assert_eq!(bhc_bs_is_prefix_of(prefix, text), 1);
        assert_eq!(bhc_bs_is_prefix_of(other, text), 0);
    }

    #[test]
    fn test_is_suffix_of() {
        let suffix = make_bs(b"World");
        let text = make_bs(b"Hello World");
        let other = make_bs(b"Hello");
        assert_eq!(bhc_bs_is_suffix_of(suffix, text), 1);
        assert_eq!(bhc_bs_is_suffix_of(other, text), 0);
    }

    #[test]
    fn test_unpack() {
        let bs = make_bs(b"Hi");
        assert_eq!(bhc_bs_unpack(bs, 0), b'H' as i64);
        assert_eq!(bhc_bs_unpack(bs, 1), b'i' as i64);
    }

    #[test]
    fn test_null_safety() {
        assert_eq!(bhc_bs_null(ptr::null()), 1);
        assert_eq!(bhc_bs_length(ptr::null()), 0);
        assert_eq!(bhc_bs_head(ptr::null()), 0);
        assert_eq!(bhc_bs_last(ptr::null()), 0);
        let _ = bhc_bs_tail(ptr::null());
        let _ = bhc_bs_init(ptr::null());
        let _ = bhc_bs_append(ptr::null(), ptr::null());
        let _ = bhc_bs_reverse(ptr::null());
    }
}
