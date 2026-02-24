//! Lazy Text FFI primitives for BHC Data.Text.Lazy.
//!
//! Lazy Text is represented as a cons-list of strict Text chunks using
//! BHC's standard ADT layout:
//!
//! # Representation
//!
//! ```text
//! Tag 0 (Empty): [i64 tag=0]                                      (8 bytes)
//! Tag 1 (Chunk): [i64 tag=1][ptr strict_chunk][ptr rest]           (24 bytes)
//! ```
//!
//! This matches GHC's `Empty | Chunk !StrictText LazyText` representation.

use std::alloc::{self, Layout};

use crate::text::{alloc_text_from_bytes, text_bytes};

// ============================================================
// Lazy ADT layout helpers
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

/// Collect all strict chunk byte slices from a lazy text into a Vec.
unsafe fn collect_chunks(mut ptr: *const u8) -> Vec<&'static [u8]> {
    let mut chunks = Vec::new();
    while !ptr.is_null() && lazy_tag(ptr) == LAZY_CHUNK_TAG {
        let chunk = lazy_chunk(ptr);
        if !chunk.is_null() {
            let bytes = text_bytes(chunk);
            if !bytes.is_empty() {
                chunks.push(bytes);
            }
        }
        ptr = lazy_rest(ptr);
    }
    chunks
}

// ============================================================
// Creation
// ============================================================

/// Return an empty lazy Text.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_empty() -> *mut u8 {
    alloc_lazy_empty()
}

/// Wrap a strict Text in a single-chunk lazy Text.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_from_strict(strict: *const u8) -> *mut u8 {
    if strict.is_null() {
        return alloc_lazy_empty();
    }
    let empty = alloc_lazy_empty();
    alloc_lazy_chunk(strict as *mut u8, empty)
}

/// Flatten a lazy Text to a strict Text by concatenating all chunks.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_to_strict(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_text_from_bytes(&[]);
    }
    unsafe {
        let chunks = collect_chunks(lazy);
        if chunks.is_empty() {
            return alloc_text_from_bytes(&[]);
        }
        if chunks.len() == 1 {
            return alloc_text_from_bytes(chunks[0]);
        }
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut combined = Vec::with_capacity(total);
        for chunk in &chunks {
            combined.extend_from_slice(chunk);
        }
        alloc_text_from_bytes(&combined)
    }
}

/// Pack a cons-list of Char into a lazy Text (single chunk).
#[no_mangle]
pub extern "C" fn bhc_lazy_text_pack(charlist: *const u8) -> *mut u8 {
    // Reuse strict pack, then wrap
    let strict = crate::text::bhc_text_pack(charlist);
    if strict.is_null() {
        return alloc_lazy_empty();
    }
    let empty = alloc_lazy_empty();
    alloc_lazy_chunk(strict, empty)
}

/// Unpack a lazy Text to a cons-list of Char.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_unpack(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_nil();
    }
    unsafe {
        // Collect all bytes, interpret as UTF-8, build char cons-list
        let chunks = collect_chunks(lazy);
        let mut all_bytes = Vec::new();
        for chunk in &chunks {
            all_bytes.extend_from_slice(chunk);
        }
        let s = std::str::from_utf8_unchecked(&all_bytes);
        // Build cons-list from end
        let chars: Vec<char> = s.chars().collect();
        let mut list = alloc_nil();
        for &c in chars.iter().rev() {
            let char_val = c as i64;
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

// ============================================================
// Queries
// ============================================================

/// Check if a lazy Text is empty. Returns 1 if empty, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_null(lazy: *const u8) -> i64 {
    if lazy.is_null() {
        return 1;
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return 1;
        }
        // Check if the chunk is empty and the rest is also empty
        let chunk = lazy_chunk(lazy);
        if chunk.is_null() || text_bytes(chunk).is_empty() {
            let rest = lazy_rest(lazy);
            return bhc_lazy_text_null(rest);
        }
        0
    }
}

/// Get the total character count of a lazy Text.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_length(lazy: *const u8) -> i64 {
    if lazy.is_null() {
        return 0;
    }
    unsafe {
        let chunks = collect_chunks(lazy);
        let mut total: i64 = 0;
        for chunk in &chunks {
            // Count UTF-8 characters
            let s = std::str::from_utf8_unchecked(chunk);
            total += s.chars().count() as i64;
        }
        total
    }
}

// ============================================================
// Operations
// ============================================================

/// Append two lazy Texts.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_append(a: *const u8, b: *const u8) -> *mut u8 {
    if a.is_null() || (unsafe { lazy_tag(a) } == LAZY_EMPTY_TAG) {
        if b.is_null() {
            return alloc_lazy_empty();
        }
        return b as *mut u8;
    }
    if b.is_null() || (unsafe { lazy_tag(b) } == LAZY_EMPTY_TAG) {
        return a as *mut u8;
    }
    // Walk a's chunks, rebuild with b at the end
    unsafe { append_chunks(a, b) }
}

/// Recursively rebuild chunk list a with b appended at the end.
unsafe fn append_chunks(a: *const u8, b: *const u8) -> *mut u8 {
    if a.is_null() || lazy_tag(a) == LAZY_EMPTY_TAG {
        return b as *mut u8;
    }
    let chunk = lazy_chunk(a);
    let rest = lazy_rest(a);
    let new_rest = append_chunks(rest, b);
    alloc_lazy_chunk(chunk as *mut u8, new_rest)
}

/// Convert a BHC cons-list of strict Text values to a lazy Text.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_from_chunks(list: *const u8) -> *mut u8 {
    if list.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        let tag = *(list as *const i64);
        if tag == 0 {
            return alloc_lazy_empty();
        }
        // Cons: tag=1, head at offset 8, tail at offset 16
        let head = *((list as *const *const u8).add(1));
        let tail = *((list as *const *const u8).add(2));
        let rest = bhc_lazy_text_from_chunks(tail);
        alloc_lazy_chunk(head as *mut u8, rest)
    }
}

/// Convert a lazy Text to a BHC cons-list of strict Text values.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_to_chunks(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_nil();
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return alloc_nil();
        }
        let chunk = lazy_chunk(lazy);
        let rest = lazy_rest(lazy);
        let tail_list = bhc_lazy_text_to_chunks(rest);
        alloc_cons(chunk as *mut u8, tail_list)
    }
}

/// Get the first character of a lazy Text.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_head(lazy: *const u8) -> i64 {
    if lazy.is_null() {
        return 0;
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return 0;
        }
        let chunk = lazy_chunk(lazy);
        if chunk.is_null() || text_bytes(chunk).is_empty() {
            return bhc_lazy_text_head(lazy_rest(lazy));
        }
        crate::text::bhc_text_head(chunk)
    }
}

/// Get the tail of a lazy Text (everything after the first character).
#[no_mangle]
pub extern "C" fn bhc_lazy_text_tail(lazy: *const u8) -> *mut u8 {
    if lazy.is_null() {
        return alloc_lazy_empty();
    }
    unsafe {
        if lazy_tag(lazy) == LAZY_EMPTY_TAG {
            return alloc_lazy_empty();
        }
        let chunk = lazy_chunk(lazy);
        let rest = lazy_rest(lazy);
        if chunk.is_null() || text_bytes(chunk).is_empty() {
            return bhc_lazy_text_tail(rest);
        }
        let tail_chunk = crate::text::bhc_text_tail(chunk);
        // If the tail chunk is empty, return rest
        if tail_chunk.is_null() || text_bytes(tail_chunk).is_empty() {
            return rest as *mut u8;
        }
        alloc_lazy_chunk(tail_chunk, rest as *mut u8)
    }
}

/// Take the first n characters of a lazy Text.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_take(n: i64, lazy: *const u8) -> *mut u8 {
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
            return bhc_lazy_text_take(n, rest);
        }
        let bytes = text_bytes(chunk);
        let s = std::str::from_utf8_unchecked(bytes);
        let chunk_chars = s.chars().count() as i64;
        if n >= chunk_chars {
            // Take the whole chunk and continue
            let rest_taken = bhc_lazy_text_take(n - chunk_chars, rest);
            alloc_lazy_chunk(chunk as *mut u8, rest_taken)
        } else {
            // Take partial chunk
            let take_bytes: usize = s.chars().take(n as usize).map(|c| c.len_utf8()).sum();
            let taken = alloc_text_from_bytes(&bytes[..take_bytes]);
            let empty = alloc_lazy_empty();
            alloc_lazy_chunk(taken, empty)
        }
    }
}

/// Drop the first n characters of a lazy Text.
#[no_mangle]
pub extern "C" fn bhc_lazy_text_drop(n: i64, lazy: *const u8) -> *mut u8 {
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
            return bhc_lazy_text_drop(n, rest);
        }
        let bytes = text_bytes(chunk);
        let s = std::str::from_utf8_unchecked(bytes);
        let chunk_chars = s.chars().count() as i64;
        if n >= chunk_chars {
            // Drop entire chunk
            bhc_lazy_text_drop(n - chunk_chars, rest)
        } else {
            // Drop partial chunk
            let drop_bytes: usize = s.chars().take(n as usize).map(|c| c.len_utf8()).sum();
            let remaining = alloc_text_from_bytes(&bytes[drop_bytes..]);
            alloc_lazy_chunk(remaining, rest as *mut u8)
        }
    }
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

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_text_empty() {
        let e = bhc_lazy_text_empty();
        assert!(!e.is_null());
        assert_eq!(bhc_lazy_text_null(e), 1);
        assert_eq!(bhc_lazy_text_length(e), 0);
    }

    #[test]
    fn test_lazy_text_from_strict_roundtrip() {
        let strict = alloc_text_from_bytes(b"hello");
        let lazy = bhc_lazy_text_from_strict(strict);
        assert_eq!(bhc_lazy_text_null(lazy), 0);
        assert_eq!(bhc_lazy_text_length(lazy), 5);
        let back = bhc_lazy_text_to_strict(lazy);
        unsafe {
            assert_eq!(text_bytes(back), b"hello");
        }
    }

    #[test]
    fn test_lazy_text_append() {
        let a = bhc_lazy_text_from_strict(alloc_text_from_bytes(b"hello "));
        let b = bhc_lazy_text_from_strict(alloc_text_from_bytes(b"world"));
        let combined = bhc_lazy_text_append(a, b);
        let strict = bhc_lazy_text_to_strict(combined);
        unsafe {
            assert_eq!(text_bytes(strict), b"hello world");
        }
    }

    #[test]
    fn test_lazy_text_head_tail() {
        let lazy = bhc_lazy_text_from_strict(alloc_text_from_bytes(b"abc"));
        assert_eq!(bhc_lazy_text_head(lazy), b'a' as i64);
        let tail = bhc_lazy_text_tail(lazy);
        assert_eq!(bhc_lazy_text_head(tail), b'b' as i64);
        let strict = bhc_lazy_text_to_strict(tail);
        unsafe {
            assert_eq!(text_bytes(strict), b"bc");
        }
    }

    #[test]
    fn test_lazy_text_take_drop() {
        let lazy = bhc_lazy_text_from_strict(alloc_text_from_bytes(b"hello world"));
        let taken = bhc_lazy_text_take(5, lazy);
        let strict_taken = bhc_lazy_text_to_strict(taken);
        unsafe {
            assert_eq!(text_bytes(strict_taken), b"hello");
        }
        let dropped = bhc_lazy_text_drop(6, lazy);
        let strict_dropped = bhc_lazy_text_to_strict(dropped);
        unsafe {
            assert_eq!(text_bytes(strict_dropped), b"world");
        }
    }
}
