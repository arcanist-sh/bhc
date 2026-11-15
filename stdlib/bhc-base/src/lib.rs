//! BHC Base Library - Rust support
//!
//! This crate provides minimal Rust support for the BHC base library.
//!
//! # Architecture
//!
//! The BHC base library is **implemented in Haskell** (see `hs/BHC/Data/*.hs`
//! and `hs/BHC/Control/*.hs`). BHC compiles the Haskell source directly.
//!
//! This Rust crate only provides:
//! - Character primitives (Unicode operations via Rust's char handling)
//! - Re-exports from bhc-rts
//!
//! # What does NOT belong here
//!
//! - List, Monad, Applicative, Arrow implementations
//! - Those are Haskell and compiled by BHC

#![warn(missing_docs)]
#![allow(unsafe_code)]

pub mod char; // Unicode primitives - actually needed
pub mod list; // List RTS operations (sort, nub, group, etc.)
pub mod string; // String RTS operations (lines, words, etc.)

/// Tell this library how the program holds scalars in value words: nonzero
/// for boxed (#94). Called once by the compiled program's `main`, before
/// any Haskell code runs. Each runtime library has its own copy of the
/// switch.
#[no_mangle]
pub extern "C" fn bhc_base_set_value_rep(boxed: i32) {
    bhc_rts_alloc::object::set_boxed(boxed != 0);
}
