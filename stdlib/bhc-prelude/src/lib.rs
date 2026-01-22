//! BHC Prelude - Rust support for stdlib primitives
//!
//! This crate provides minimal Rust support for the BHC standard library.
//!
//! # Architecture
//!
//! The BHC standard library is **implemented in Haskell** (see `hs/BHC/Prelude.hs`).
//! BHC compiles the Haskell source directly to native code.
//!
//! This Rust crate only provides:
//! - Re-exports from `bhc-rts` for primitive operations
//! - FFI glue that cannot be expressed in Haskell
//!
//! # What belongs here
//!
//! - Primitive numeric operations (Int#, Float# intrinsics)
//! - Error/panic handling primitives
//! - Re-exports from bhc-rts
//!
//! # What does NOT belong here
//!
//! - Implementations of Maybe, Either, Bool, List, etc.
//! - Those are Haskell types defined in `hs/` and compiled by BHC

#![warn(missing_docs)]
#![allow(unsafe_code)]

// This crate is intentionally minimal.
// The real Prelude implementation is in hs/BHC/Prelude.hs
