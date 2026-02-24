//! ByteString Builder FFI primitives for BHC Data.ByteString.Builder.
//!
//! Builder is represented as a **chunk-list of strict ByteStrings** — the same
//! ADT layout as lazy ByteString (`Empty | Chunk !BS LazyBS`). Many Builder
//! operations reuse existing lazy ByteString RTS functions directly via codegen
//! dispatch (empty, append, fromStrict, toStrict, hPutStr).
//!
//! This module provides the construction functions that create new chunks:
//! singleton, charUtf8, stringUtf8, intDec, binary encodings, hex, etc.

use crate::bytestring::alloc_bs_from_bytes;
use crate::lazy_bytestring::{alloc_lazy_chunk, alloc_lazy_empty};

// ============================================================
// BHC cons-list helpers (for walking Char lists)
// ============================================================

/// Extract characters from a BHC cons-list of Char, returning a String.
unsafe fn chars_from_list(mut ptr: *const u8) -> String {
    let mut result = String::new();
    loop {
        if ptr.is_null() {
            break;
        }
        let tag = *(ptr as *const i64);
        if tag == 0 {
            break; // Nil
        }
        let fields_base = (ptr as *const *const u8).add(1);
        let head_raw = *fields_base;
        let codepoint = head_raw as i64;
        if let Some(c) = char::from_u32(codepoint as u32) {
            result.push(c);
        }
        ptr = *fields_base.add(1);
    }
    result
}

// ============================================================
// Core Construction (VarIds 1000478-1000482)
// ============================================================

/// Create a Builder containing a single byte (Word8).
/// VarId 1000478
#[no_mangle]
pub extern "C" fn bhc_bsb_singleton(byte: i64) -> *mut u8 {
    let bs = alloc_bs_from_bytes(&[byte as u8]);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a single UTF-8 encoded character.
/// VarId 1000479
#[no_mangle]
pub extern "C" fn bhc_bsb_char_utf8(codepoint: i64) -> *mut u8 {
    let mut buf = [0u8; 4];
    let bytes = match char::from_u32(codepoint as u32) {
        Some(c) => c.encode_utf8(&mut buf).as_bytes(),
        None => {
            // Replacement character U+FFFD
            '\u{FFFD}'.encode_utf8(&mut buf).as_bytes()
        }
    };
    let bs = alloc_bs_from_bytes(bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder from a Haskell String (cons-list of Char), UTF-8 encoded.
/// VarId 1000480
#[no_mangle]
pub extern "C" fn bhc_bsb_string_utf8(list_ptr: *mut u8) -> *mut u8 {
    let s = unsafe { chars_from_list(list_ptr) };
    let bs = alloc_bs_from_bytes(s.as_bytes());
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing the decimal ASCII representation of an Int.
/// VarId 1000481
#[no_mangle]
pub extern "C" fn bhc_bsb_int_dec(value: i64) -> *mut u8 {
    let s = value.to_string();
    let bs = alloc_bs_from_bytes(s.as_bytes());
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a single ASCII byte (codepoint masked to 7 bits).
/// VarId 1000482
#[no_mangle]
pub extern "C" fn bhc_bsb_char7(codepoint: i64) -> *mut u8 {
    let byte = (codepoint & 0x7f) as u8;
    let bs = alloc_bs_from_bytes(&[byte]);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

// ============================================================
// Encoding Functions (VarIds 1000483-1000490)
// ============================================================

/// Create a Builder containing a single Latin-1 byte (codepoint masked to 8 bits).
/// VarId 1000483
#[no_mangle]
pub extern "C" fn bhc_bsb_char8(codepoint: i64) -> *mut u8 {
    let byte = (codepoint & 0xff) as u8;
    let bs = alloc_bs_from_bytes(&[byte]);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder from a Haskell String, ASCII-encoded (masked to 7 bits).
/// VarId 1000484
#[no_mangle]
pub extern "C" fn bhc_bsb_string7(list_ptr: *mut u8) -> *mut u8 {
    let s = unsafe { chars_from_list(list_ptr) };
    let bytes: Vec<u8> = s.bytes().map(|b| b & 0x7f).collect();
    let bs = alloc_bs_from_bytes(&bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder from a Haskell String, Latin-1 encoded (masked to 8 bits).
/// VarId 1000485
#[no_mangle]
pub extern "C" fn bhc_bsb_string8(list_ptr: *mut u8) -> *mut u8 {
    let s = unsafe { chars_from_list(list_ptr) };
    let bytes: Vec<u8> = s.chars().map(|c| (c as u32 & 0xff) as u8).collect();
    let bs = alloc_bs_from_bytes(&bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a Word16 in big-endian byte order.
/// VarId 1000486
#[no_mangle]
pub extern "C" fn bhc_bsb_word16_be(value: i64) -> *mut u8 {
    let bytes = (value as u16).to_be_bytes();
    let bs = alloc_bs_from_bytes(&bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a Word32 in big-endian byte order.
/// VarId 1000487
#[no_mangle]
pub extern "C" fn bhc_bsb_word32_be(value: i64) -> *mut u8 {
    let bytes = (value as u32).to_be_bytes();
    let bs = alloc_bs_from_bytes(&bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a Word64 in big-endian byte order.
/// VarId 1000488
#[no_mangle]
pub extern "C" fn bhc_bsb_word64_be(value: i64) -> *mut u8 {
    let bytes = (value as u64).to_be_bytes();
    let bs = alloc_bs_from_bytes(&bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a Word16 in little-endian byte order.
/// VarId 1000489
#[no_mangle]
pub extern "C" fn bhc_bsb_word16_le(value: i64) -> *mut u8 {
    let bytes = (value as u16).to_le_bytes();
    let bs = alloc_bs_from_bytes(&bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a Word32 in little-endian byte order.
/// VarId 1000490
#[no_mangle]
pub extern "C" fn bhc_bsb_word32_le(value: i64) -> *mut u8 {
    let bytes = (value as u32).to_le_bytes();
    let bs = alloc_bs_from_bytes(&bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

// ============================================================
// More Encoding (VarIds 1000491-1000493)
// ============================================================
// Note: bhc_float_to_word32 and bhc_double_to_word64 are in bytearray.rs

/// Create a Builder containing a Word64 in little-endian byte order.
/// VarId 1000491
#[no_mangle]
pub extern "C" fn bhc_bsb_word64_le(value: i64) -> *mut u8 {
    let bytes = (value as u64).to_le_bytes();
    let bs = alloc_bs_from_bytes(&bytes);
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a variable-length lowercase hex encoding of a Word.
/// VarId 1000492
#[no_mangle]
pub extern "C" fn bhc_bsb_word_hex(value: i64) -> *mut u8 {
    let s = format!("{:x}", value as u64);
    let bs = alloc_bs_from_bytes(s.as_bytes());
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

/// Create a Builder containing a fixed 2-digit lowercase hex encoding of a Word8.
/// VarId 1000493
#[no_mangle]
pub extern "C" fn bhc_bsb_word8_hex_fixed(value: i64) -> *mut u8 {
    let s = format!("{:02x}", (value & 0xff) as u8);
    let bs = alloc_bs_from_bytes(s.as_bytes());
    alloc_lazy_chunk(bs, alloc_lazy_empty())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytestring::bs_bytes;
    use crate::lazy_bytestring::{lazy_chunk, lazy_rest, lazy_tag};

    /// Helper: extract all bytes from a Builder (lazy BS chunk list).
    unsafe fn builder_bytes(ptr: *const u8) -> Vec<u8> {
        let mut result = Vec::new();
        let mut cur = ptr;
        while !cur.is_null() && lazy_tag(cur) == 1 {
            let chunk = lazy_chunk(cur);
            if !chunk.is_null() {
                result.extend_from_slice(bs_bytes(chunk));
            }
            cur = lazy_rest(cur);
        }
        result
    }

    #[test]
    fn test_singleton() {
        let b = bhc_bsb_singleton(65); // 'A'
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, vec![65]);
    }

    #[test]
    fn test_char_utf8_ascii() {
        let b = bhc_bsb_char_utf8(65); // 'A'
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, vec![65]);
    }

    #[test]
    fn test_char_utf8_multibyte() {
        let b = bhc_bsb_char_utf8(0x00E9); // 'é'
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, vec![0xC3, 0xA9]);
    }

    #[test]
    fn test_int_dec_positive() {
        let b = bhc_bsb_int_dec(12345);
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, b"12345");
    }

    #[test]
    fn test_int_dec_negative() {
        let b = bhc_bsb_int_dec(-42);
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, b"-42");
    }

    #[test]
    fn test_word16_be() {
        let b = bhc_bsb_word16_be(0x0102);
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, vec![0x01, 0x02]);
    }

    #[test]
    fn test_word16_le() {
        let b = bhc_bsb_word16_le(0x0102);
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, vec![0x02, 0x01]);
    }

    #[test]
    fn test_word_hex() {
        let b = bhc_bsb_word_hex(255);
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, b"ff");
    }

    #[test]
    fn test_word8_hex_fixed() {
        let b = bhc_bsb_word8_hex_fixed(10);
        let bytes = unsafe { builder_bytes(b) };
        assert_eq!(bytes, b"0a");
    }

}
