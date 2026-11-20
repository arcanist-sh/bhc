//! BHC Text Library - Rust support
//!
//! This crate provides SIMD-accelerated primitives for text processing.
//!
//! # Architecture
//!
//! Text and ByteString types are **implemented in Haskell** (see
//! `hs/BHC/Data/Text.hs` and `hs/BHC/Data/ByteString.hs`). BHC compiles
//! them directly.
//!
//! This Rust crate provides:
//! - SIMD-accelerated search operations
//! - SIMD-accelerated case conversion
//! - UTF-8 validation primitives
//! - Pinned memory for FFI interop
//!
//! # What belongs here
//!
//! - Low-level SIMD intrinsics for text search (memchr, find, etc.)
//! - SIMD-accelerated case conversion (toLower, toUpper)
//! - UTF-8 validation and scanning
//! - Pinned buffer management for FFI
//!
//! # What does NOT belong here
//!
//! - Full Text/ByteString type definitions (those are Haskell)
//! - High-level string operations (map, filter, etc.)
//! - Those are Haskell and compiled by BHC

#![warn(missing_docs)]
#![allow(unsafe_code)] // SIMD requires unsafe

pub mod builder;
pub mod bytearray;
pub mod bytestring;
pub mod io_error;
pub mod lazy_bytestring;
pub mod lazy_text;
pub mod simd;
pub mod text;
pub mod text_io;

/// Tell this library how the program holds scalars in value words: nonzero
/// for boxed (#94). Called once by the compiled program's `main`, before
/// any Haskell code runs. Each runtime library has its own copy of the
/// switch.
#[no_mangle]
pub extern "C" fn bhc_text_set_value_rep(boxed: i32) {
    bhc_rts_alloc::object::set_boxed(boxed != 0);
}
