//! Lazy ByteString FFI primitives for BHC Data.ByteString.Lazy,
//! Data.ByteString.Lazy.Char8, and Data.Text.Lazy.Encoding.
//!
//! Lazy ByteString is represented as a cons-list of strict ByteString chunks
//! using BHC's standard ADT layout:
//!
//! # Representation
//!
//! ```text
//! Tag 0 (Empty): [i64 tag=0]                                      (8 bytes)
//! Tag 1 (Chunk): [i64 tag=1][ptr strict_chunk][ptr rest]           (24 bytes)
//! ```
//!
//! This matches GHC's `Empty | Chunk !StrictByteString LazyByteString`.

use std::alloc::{self, Layout};
use std::io::{Read, Write};

use crate::bytestring::{alloc_bs_from_bytes, bs_bytes};
use crate::text::{alloc_text_from_bytes, text_bytes};

// ============================================================
// Lazy ADT layout helpers (shared with lazy_text.rs)
// ============================================================

const LAZY_EMPTY_TAG: i64 = 0;
const LAZY_CHUNK_TAG: i64 = 1;

/// Read the tag from a lazy value.
unsafe fn lazy_tag(ptr: *const u8) -> i64 {
    *(ptr as *const i64)
}

/// Read the strict chunk pointer from a Chunk node (at offset 8).
unsafe fn lazy_chunk(ptr: *const u8) -> *const u8 {
    *((ptr as *const *const u8).add(1))
}

/// Read the rest pointer from a Chunk node (at offset 16).
unsafe fn lazy_rest(ptr: *const u8) -> *const u8 {
    *((ptr as *const *const u8).add(2))
}

/// Allocate an Empty lazy node (tag=0, 8 bytes).
fn alloc_lazy_empty() -> *mut u8 {
    let layout = Layout::from_size_align(8, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc(layout);
        (ptr as *mut i64).write(LAZY_EMPTY_TAG);
        ptr
    }
}

/// Allocate a Chunk lazy node (tag=1, 24 bytes).
fn alloc_lazy_chunk(chunk: *mut u8, rest: *mut u8) -> *mut u8 {
    let layout = Layout::from_size_align(24, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc(layout);
        (ptr as *mut i64).write(LAZY_CHUNK_TAG);
        (ptr as *mut *mut u8).add(1).write(chunk);
        (ptr as *mut *mut u8).add(2).write(rest);
        ptr
    }
}

/// Collect all strict chunk byte slices from a lazy ByteString.
unsafe fn collect_chunks(mut ptr: *const u8) -> Vec<&'static [u8]> {
    let mut chunks = Vec::new();
    while !ptr.is_null() && lazy_tag(ptr) == LAZY_CHUNK_TAG {
        let chunk = lazy_chunk(ptr);
        if !chunk.is_null() {
            let bytes = bs_bytes(chunk);
            if !bytes.is_empty() {
                chunks.push(bytes);
            }
        }
        ptr = lazy_rest(ptr);
    }
    chunks
}

/// Extract a path string from a BHC cons-list of Char.
unsafe fn extract_path(path_ptr: *const u8) -> String {
    let mut path_str = String::new();
    let mut current = path_ptr;
    loop {
        if current.is_null() {
            break;
        }
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
    }
    path_str
}

/// Concatenate all chunks into a single byte vector.
unsafe fn flatten_to_bytes(ptr: *const u8) -> Vec<u8> {
    let chunks = collect_chunks(ptr);
    let total: usize = chunks.iter().map(|c| c.len()).sum();
    let mut combined = Vec::with_capacity(total);
    for chunk in &chunks {
        combined.extend_from_slice(chunk);
    }
    combined
}

// ============================================================
// BHC cons-list helpers
// ============================================================

/// Allocate a Nil node (tag=0, 8 bytes).
fn alloc_nil() -> *mut u8 {
    let layout = Layout::from_size_align(8, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc(layout);
        (ptr as *mut i64).write(0);
        ptr
    }
}

/// Allocate a Cons node (tag=1, head, tail — 24 bytes).
fn alloc_cons(head: *mut u8, tail: *mut u8) -> *mut u8 {
    let layout = Layout::from_size_align(24, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc(layout);
        (ptr as *mut i64).write(1);
        (ptr as *mut *mut u8).add(1).write(head);
        (ptr as *mut *mut u8).add(2).write(tail);
        ptr
    }
}

// Handle sentinel constants (matching bhc-rts)
const HANDLE_STDIN: usize = 1;
const HANDLE_STDOUT: usize = 2;
const HANDLE_STDERR: usize = 3;

/// Mirror of BhcHandle from bhc-rts.
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
// Data.ByteString.Lazy — Creation
// ============================================================

/// Return an empty lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_empty() -> *mut u8 {
    alloc_lazy_empty()
}

/// Wrap a strict ByteString in a single-chunk lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_from_strict(strict: *const u8) -> *mut u8 {
    if strict.is_null() {
        return alloc_lazy_empty();
    }
    let empty = alloc_lazy_empty();
    alloc_lazy_chunk(strict as *mut u8, empty)
}

/// Flatten a lazy ByteString to a strict ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_to_strict(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_bs_from_bytes(&[]);
    }
    unsafe {
        let chunks = collect_chunks(lazy);
        if chunks.is_empty() {
            return alloc_bs_from_bytes(&[]);
        }
        if chunks.len() == 1 {
            return alloc_bs_from_bytes(chunks[0]);
        }
        let combined = flatten_to_bytes(lazy);
        alloc_bs_from_bytes(&combined)
    }
}

/// Convert a BHC cons-list of strict ByteString values to lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_from_chunks(list: *const u8) -> *mut u8 {
    if list.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        let tag = *(list as *const i64);
        if tag == 0 {
            return alloc_lazy_empty();
        }
        let head = *((list as *const *const u8).add(1));
        let tail = *((list as *const *const u8).add(2));
        let rest = bhc_lazy_bs_from_chunks(tail);
        alloc_lazy_chunk(head as *mut u8, rest)
    }
}

/// Convert a lazy ByteString to a BHC cons-list of strict ByteString values.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_to_chunks(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_nil();
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return alloc_nil();
        }
        let chunk = lazy_chunk(lazy);
        let rest = lazy_rest(lazy);
        let tail_list = bhc_lazy_bs_to_chunks(rest);
        alloc_cons(chunk as *mut u8, tail_list)
    }
}

// ============================================================
// Data.ByteString.Lazy — Queries
// ============================================================

/// Check if a lazy ByteString is empty. Returns 1 if empty, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_null(lazy: *const u8) -> i64 {
    if lazy.is_null() {
        return 1;
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return 1;
        }
        let chunk = lazy_chunk(lazy);
        if chunk.is_null() || bs_bytes(chunk).is_empty() {
            return bhc_lazy_bs_null(lazy_rest(lazy));
        }
        0
    }
}

/// Get the total byte length of a lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_length(lazy: *const u8) -> i64 {
    if lazy.is_null() {
        return 0;
    }
    unsafe {
        let chunks = collect_chunks(lazy);
        chunks.iter().map(|c| c.len() as i64).sum()
    }
}

/// Pack a cons-list of Word8 (as i64) into a lazy ByteString (single chunk).
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_pack(list: *const u8) -> *mut u8 {
    if list.is_null() {
        return alloc_lazy_empty();
    }
    // Walk cons-list, collect bytes
    let mut bytes = Vec::new();
    let mut current = list;
    unsafe {
        loop {
            if current.is_null() {
                break;
            }
            let tag = *(current as *const i64);
            if tag == 0 {
                break;
            }
            let head = *((current as *const *const u8).add(1));
            bytes.push(head as u8);
            current = *((current as *const *const u8).add(2));
        }
    }
    let strict = alloc_bs_from_bytes(&bytes);
    let empty = alloc_lazy_empty();
    alloc_lazy_chunk(strict, empty)
}

// ============================================================
// Data.ByteString.Lazy — Operations
// ============================================================

/// Append two lazy ByteStrings.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_append(a: *const u8, b: *const u8) -> *mut u8 {
    if a.is_null() || (unsafe { lazy_tag(a) } == LAZY_EMPTY_TAG) {
        if b.is_null() {
            return alloc_lazy_empty();
        }
        return b as *mut u8;
    }
    if b.is_null() || (unsafe { lazy_tag(b) } == LAZY_EMPTY_TAG) {
        return a as *mut u8;
    }
    unsafe { append_chunks(a, b) }
}

unsafe fn append_chunks(a: *const u8, b: *const u8) -> *mut u8 {
    if a.is_null() || lazy_tag(a) == LAZY_EMPTY_TAG {
        return b as *mut u8;
    }
    let chunk = lazy_chunk(a);
    let rest = lazy_rest(a);
    let new_rest = append_chunks(rest, b);
    alloc_lazy_chunk(chunk as *mut u8, new_rest)
}

/// Get the first byte of a lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_head(lazy: *const u8) -> i64 {
    if lazy.is_null() {
        return 0;
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return 0;
        }
        let chunk = lazy_chunk(lazy);
        if chunk.is_null() || bs_bytes(chunk).is_empty() {
            return bhc_lazy_bs_head(lazy_rest(lazy));
        }
        bs_bytes(chunk)[0] as i64
    }
}

/// Get the tail of a lazy ByteString (drop first byte).
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_tail(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return alloc_lazy_empty();
        }
        let chunk = lazy_chunk(lazy);
        let rest = lazy_rest(lazy);
        if chunk.is_null() || bs_bytes(chunk).is_empty() {
            return bhc_lazy_bs_tail(rest);
        }
        let bytes = bs_bytes(chunk);
        if bytes.len() <= 1 {
            return rest as *mut u8;
        }
        let tail_chunk = alloc_bs_from_bytes(&bytes[1..]);
        alloc_lazy_chunk(tail_chunk, rest as *mut u8)
    }
}

/// Take the first n bytes of a lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_take(n: i64, lazy: *const u8) -> *mut u8 {
    if n <= 0 || lazy.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return alloc_lazy_empty();
        }
        let chunk = lazy_chunk(lazy);
        let rest = lazy_rest(lazy);
        if chunk.is_null() {
            return bhc_lazy_bs_take(n, rest);
        }
        let bytes = bs_bytes(chunk);
        let chunk_len = bytes.len() as i64;
        if n >= chunk_len {
            let rest_taken = bhc_lazy_bs_take(n - chunk_len, rest);
            alloc_lazy_chunk(chunk as *mut u8, rest_taken)
        } else {
            let taken = alloc_bs_from_bytes(&bytes[..n as usize]);
            let empty = alloc_lazy_empty();
            alloc_lazy_chunk(taken, empty)
        }
    }
}

/// Drop the first n bytes of a lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_drop(n: i64, lazy: *const u8) -> *mut u8 {
    if n <= 0 {
        if lazy.is_null() {
            return alloc_lazy_empty();
        }
        return lazy as *mut u8;
    }
    if lazy.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return alloc_lazy_empty();
        }
        let chunk = lazy_chunk(lazy);
        let rest = lazy_rest(lazy);
        if chunk.is_null() {
            return bhc_lazy_bs_drop(n, rest);
        }
        let bytes = bs_bytes(chunk);
        let chunk_len = bytes.len() as i64;
        if n >= chunk_len {
            bhc_lazy_bs_drop(n - chunk_len, rest)
        } else {
            let remaining = alloc_bs_from_bytes(&bytes[n as usize..]);
            alloc_lazy_chunk(remaining, rest as *mut u8)
        }
    }
}

/// Filter a lazy ByteString with a closure predicate.
///
/// `fn_ptr(env_ptr, byte) -> i64` where non-zero means keep.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_filter(
    fn_ptr: extern "C" fn(*mut u8, i64) -> i64,
    env_ptr: *mut u8,
    lazy: *const u8,
) -> *mut u8 {
    if lazy.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        let all_bytes = flatten_to_bytes(lazy);
        let filtered: Vec<u8> = all_bytes
            .iter()
            .filter(|&&b| fn_ptr(env_ptr, b as i64) != 0)
            .copied()
            .collect();
        if filtered.is_empty() {
            return alloc_lazy_empty();
        }
        let strict = alloc_bs_from_bytes(&filtered);
        let empty = alloc_lazy_empty();
        alloc_lazy_chunk(strict, empty)
    }
}

/// Check if one lazy ByteString is a prefix of another.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_is_prefix_of(prefix: *const u8, haystack: *const u8) -> i64 {
    unsafe {
        let prefix_bytes = if prefix.is_null() {
            Vec::new()
        } else {
            flatten_to_bytes(prefix)
        };
        let haystack_bytes = if haystack.is_null() {
            Vec::new()
        } else {
            flatten_to_bytes(haystack)
        };
        if haystack_bytes.starts_with(&prefix_bytes) {
            1
        } else {
            0
        }
    }
}

// ============================================================
// Data.ByteString.Lazy — I/O
// ============================================================

/// Read a file into a lazy ByteString (single chunk).
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_read_file(path_ptr: *const u8) -> *mut u8 {
    if path_ptr.is_null() {
        return alloc_lazy_empty();
    }
    let path_str = unsafe { extract_path(path_ptr) };
    match std::fs::read(&path_str) {
        Ok(bytes) => {
            if bytes.is_empty() {
                return alloc_lazy_empty();
            }
            let strict = alloc_bs_from_bytes(&bytes);
            let empty = alloc_lazy_empty();
            alloc_lazy_chunk(strict, empty)
        }
        Err(_) => alloc_lazy_empty(),
    }
}

/// Write a lazy ByteString to a file.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_write_file(path_ptr: *const u8, lazy: *const u8) {
    if path_ptr.is_null() {
        return;
    }
    let path_str = unsafe { extract_path(path_ptr) };
    let bytes = if lazy.is_null() {
        Vec::new()
    } else {
        unsafe { flatten_to_bytes(lazy) }
    };
    let _ = std::fs::write(&path_str, &bytes);
}

/// Write a lazy ByteString to stdout.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_put_str(lazy: *const u8) {
    if lazy.is_null() {
        return;
    }
    unsafe {
        let chunks = collect_chunks(lazy);
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        for chunk in &chunks {
            let _ = lock.write_all(chunk);
        }
        let _ = lock.flush();
    }
}

/// Write a lazy ByteString to a handle.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_h_put_str(handle: *mut u8, lazy: *const u8) {
    if lazy.is_null() {
        return;
    }
    unsafe {
        let bytes = flatten_to_bytes(lazy);
        if is_sentinel(handle) {
            let h = handle as usize;
            if h == HANDLE_STDOUT {
                let stdout = std::io::stdout();
                let mut lock = stdout.lock();
                let _ = lock.write_all(&bytes);
                let _ = lock.flush();
            } else if h == HANDLE_STDERR {
                let stderr = std::io::stderr();
                let mut lock = stderr.lock();
                let _ = lock.write_all(&bytes);
                let _ = lock.flush();
            }
        } else {
            let bhc_handle = &mut *(handle as *mut BhcHandle);
            if let Some(ref mut file) = bhc_handle.file {
                let _ = file.write_all(&bytes);
                let _ = file.flush();
            }
        }
    }
}

/// Read all remaining contents from a handle into a lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_h_get_contents(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        let mut buf = Vec::new();
        if is_sentinel(handle) {
            let h = handle as usize;
            if h == HANDLE_STDIN {
                let stdin = std::io::stdin();
                let mut lock = stdin.lock();
                let _ = lock.read_to_end(&mut buf);
            }
        } else {
            let bhc_handle = &mut *(handle as *mut BhcHandle);
            if let Some(ref mut file) = bhc_handle.file {
                let _ = file.read_to_end(&mut buf);
            }
        }
        if buf.is_empty() {
            return alloc_lazy_empty();
        }
        let strict = alloc_bs_from_bytes(&buf);
        let empty = alloc_lazy_empty();
        alloc_lazy_chunk(strict, empty)
    }
}

// ============================================================
// Data.ByteString.Lazy.Char8
// ============================================================

/// Unpack a lazy ByteString to a cons-list of Char (treating bytes as Latin-1).
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_char8_unpack(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_nil();
    }
    unsafe {
        let bytes = flatten_to_bytes(lazy);
        // Build cons-list from end
        let mut list = alloc_nil();
        for &b in bytes.iter().rev() {
            // Char is stored as tagged int (codepoint as pointer)
            let char_val = b as i64;
            let layout = Layout::from_size_align(24, 8).expect("invalid layout");
            let cons = alloc::alloc(layout);
            (cons as *mut i64).write(1); // tag = 1 (Cons)
            (cons as *mut *mut u8).add(1).write(char_val as *mut u8); // head = char as ptr
            (cons as *mut *mut u8).add(2).write(list); // tail
            list = cons;
        }
        list
    }
}

/// Split a lazy ByteString on newlines, returning a cons-list of lazy ByteStrings.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_char8_lines(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_nil();
    }
    unsafe {
        let bytes = flatten_to_bytes(lazy);
        let mut lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
        // Remove trailing empty element if input ends with \n
        if lines.last().map_or(false, |l| l.is_empty()) {
            lines.pop();
        }
        // Build cons-list from end
        let mut list = alloc_nil();
        for line in lines.iter().rev() {
            let strict = alloc_bs_from_bytes(line);
            let empty = alloc_lazy_empty();
            let lazy_line = alloc_lazy_chunk(strict, empty);
            list = alloc_cons(lazy_line, list);
        }
        list
    }
}

/// Join lazy ByteStrings with newlines.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_char8_unlines(list: *const u8) -> *mut u8 {
    if list.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        let mut parts: Vec<Vec<u8>> = Vec::new();
        let mut current = list;
        loop {
            if current.is_null() {
                break;
            }
            let tag = *(current as *const i64);
            if tag == 0 {
                break;
            }
            let head = *((current as *const *const u8).add(1));
            let bytes = flatten_to_bytes(head);
            parts.push(bytes);
            current = *((current as *const *const u8).add(2));
        }
        if parts.is_empty() {
            return alloc_lazy_empty();
        }
        let mut combined = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            combined.extend_from_slice(part);
            if i < parts.len() - 1 {
                combined.push(b'\n');
            }
        }
        combined.push(b'\n');
        let strict = alloc_bs_from_bytes(&combined);
        let empty = alloc_lazy_empty();
        alloc_lazy_chunk(strict, empty)
    }
}

/// Take first n bytes (alias for bhc_lazy_bs_take, Char8 = bytes).
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_char8_take(n: i64, lazy: *const u8) -> *mut u8 {
    bhc_lazy_bs_take(n, lazy)
}

/// Drop bytes while a Char predicate holds.
///
/// `fn_ptr(env_ptr, byte_as_char) -> i64` where non-zero means continue dropping.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_char8_drop_while(
    fn_ptr: extern "C" fn(*mut u8, i64) -> i64,
    env_ptr: *mut u8,
    lazy: *const u8,
) -> *mut u8 {
    if lazy.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        let bytes = flatten_to_bytes(lazy);
        let mut skip = 0;
        for &b in &bytes {
            if fn_ptr(env_ptr, b as i64) != 0 {
                skip += 1;
            } else {
                break;
            }
        }
        if skip >= bytes.len() {
            return alloc_lazy_empty();
        }
        let strict = alloc_bs_from_bytes(&bytes[skip..]);
        let empty = alloc_lazy_empty();
        alloc_lazy_chunk(strict, empty)
    }
}

/// Prepend a single byte (as Char) to a lazy ByteString.
#[no_mangle]
pub extern "C" fn bhc_lazy_bs_char8_cons(byte: i64, lazy: *const u8) -> *mut u8 {
    let b = [byte as u8];
    let strict = alloc_bs_from_bytes(&b);
    let rest = if lazy.is_null() {
        alloc_lazy_empty()
    } else {
        lazy as *mut u8
    };
    alloc_lazy_chunk(strict, rest)
}

// ============================================================
// Data.Text.Lazy.Encoding
// ============================================================

/// Encode a lazy Text to a lazy ByteString (UTF-8).
///
/// Maps strict `encodeUtf8` per chunk (zero-copy since Text is already UTF-8).
#[no_mangle]
pub extern "C" fn bhc_lazy_text_encode_utf8(lazy_text: *const u8) -> *mut u8 {
    if lazy_text.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        if lazy_tag(lazy_text) == LAZY_EMPTY_TAG {
            return alloc_lazy_empty();
        }
        let text_chunk = lazy_chunk(lazy_text);
        let rest = lazy_rest(lazy_text);
        // Text and ByteString have the same layout, so encodeUtf8 is just
        // creating a new header pointing to the same bytes
        let bs_chunk = if text_chunk.is_null() {
            alloc_bs_from_bytes(&[])
        } else {
            let bytes = text_bytes(text_chunk);
            alloc_bs_from_bytes(bytes)
        };
        let rest_encoded = bhc_lazy_text_encode_utf8(rest);
        alloc_lazy_chunk(bs_chunk, rest_encoded)
    }
}

/// Decode a lazy ByteString to a lazy Text (UTF-8).
///
/// Maps strict `decodeUtf8` per chunk (validates UTF-8).
#[no_mangle]
pub extern "C" fn bhc_lazy_text_decode_utf8(lazy_bs: *const u8) -> *mut u8 {
    if lazy_bs.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        if lazy_tag(lazy_bs) == LAZY_EMPTY_TAG {
            return alloc_lazy_empty();
        }
        let bs_chunk = lazy_chunk(lazy_bs);
        let rest = lazy_rest(lazy_bs);
        let text_chunk = if bs_chunk.is_null() {
            alloc_text_from_bytes(&[])
        } else {
            let bytes = bs_bytes(bs_chunk);
            // Validate UTF-8; replace invalid sequences
            let s = String::from_utf8_lossy(bytes);
            alloc_text_from_bytes(s.as_bytes())
        };
        let rest_decoded = bhc_lazy_text_decode_utf8(rest);
        alloc_lazy_chunk(text_chunk, rest_decoded)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_bs_empty() {
        let e = bhc_lazy_bs_empty();
        assert!(!e.is_null());
        assert_eq!(bhc_lazy_bs_null(e), 1);
        assert_eq!(bhc_lazy_bs_length(e), 0);
    }

    #[test]
    fn test_lazy_bs_from_strict_roundtrip() {
        let strict = alloc_bs_from_bytes(b"hello");
        let lazy = bhc_lazy_bs_from_strict(strict);
        assert_eq!(bhc_lazy_bs_null(lazy), 0);
        assert_eq!(bhc_lazy_bs_length(lazy), 5);
        let back = bhc_lazy_bs_to_strict(lazy);
        unsafe {
            assert_eq!(bs_bytes(back), b"hello");
        }
    }

    #[test]
    fn test_lazy_bs_append() {
        let a = bhc_lazy_bs_from_strict(alloc_bs_from_bytes(b"hello "));
        let b = bhc_lazy_bs_from_strict(alloc_bs_from_bytes(b"world"));
        let combined = bhc_lazy_bs_append(a, b);
        let strict = bhc_lazy_bs_to_strict(combined);
        unsafe {
            assert_eq!(bs_bytes(strict), b"hello world");
        }
    }

    #[test]
    fn test_lazy_bs_head_tail() {
        let lazy = bhc_lazy_bs_from_strict(alloc_bs_from_bytes(b"abc"));
        assert_eq!(bhc_lazy_bs_head(lazy), b'a' as i64);
        let tail = bhc_lazy_bs_tail(lazy);
        assert_eq!(bhc_lazy_bs_head(tail), b'b' as i64);
        let strict = bhc_lazy_bs_to_strict(tail);
        unsafe {
            assert_eq!(bs_bytes(strict), b"bc");
        }
    }

    #[test]
    fn test_lazy_bs_take_drop() {
        let lazy = bhc_lazy_bs_from_strict(alloc_bs_from_bytes(b"hello world"));
        let taken = bhc_lazy_bs_take(5, lazy);
        let strict_taken = bhc_lazy_bs_to_strict(taken);
        unsafe {
            assert_eq!(bs_bytes(strict_taken), b"hello");
        }
        let dropped = bhc_lazy_bs_drop(6, lazy);
        let strict_dropped = bhc_lazy_bs_to_strict(dropped);
        unsafe {
            assert_eq!(bs_bytes(strict_dropped), b"world");
        }
    }

    #[test]
    fn test_lazy_bs_is_prefix_of() {
        let prefix = bhc_lazy_bs_from_strict(alloc_bs_from_bytes(b"hel"));
        let full = bhc_lazy_bs_from_strict(alloc_bs_from_bytes(b"hello"));
        assert_eq!(bhc_lazy_bs_is_prefix_of(prefix, full), 1);
        let not_prefix = bhc_lazy_bs_from_strict(alloc_bs_from_bytes(b"xyz"));
        assert_eq!(bhc_lazy_bs_is_prefix_of(not_prefix, full), 0);
    }

    #[test]
    fn test_lazy_bs_char8_cons() {
        let lazy = bhc_lazy_bs_from_strict(alloc_bs_from_bytes(b"ello"));
        let result = bhc_lazy_bs_char8_cons(b'h' as i64, lazy);
        let strict = bhc_lazy_bs_to_strict(result);
        unsafe {
            assert_eq!(bs_bytes(strict), b"hello");
        }
    }

    #[test]
    fn test_lazy_encode_decode_roundtrip() {
        let text_strict = alloc_text_from_bytes(b"hello utf8");
        let lazy_text = {
            let empty = alloc_lazy_empty();
            alloc_lazy_chunk(text_strict, empty)
        };
        let lazy_bs = bhc_lazy_text_encode_utf8(lazy_text);
        assert_eq!(bhc_lazy_bs_length(lazy_bs), 10);
        let _lazy_text_back = bhc_lazy_text_decode_utf8(lazy_bs);
        // Flatten both and compare
        let strict_bs = bhc_lazy_bs_to_strict(lazy_bs);
        unsafe {
            assert_eq!(bs_bytes(strict_bs), b"hello utf8");
        }
    }
}
