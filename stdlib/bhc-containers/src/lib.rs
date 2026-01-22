//! BHC Containers Library - Rust support
//!
//! This crate provides minimal Rust support for BHC containers.
//!
//! # Architecture
//!
//! Container types (Map, Set, IntMap, IntSet, Sequence) are **implemented
//! in Haskell** (see `hs/BHC/Data/*.hs`). BHC compiles them directly.
//!
//! These are classic functional data structures:
//! - Weight-balanced trees (Map, Set)
//! - Patricia tries (IntMap, IntSet)
//! - Finger trees (Sequence)
//!
//! This Rust crate is intentionally empty. All functionality comes from
//! the Haskell source.

#![warn(missing_docs)]
#![allow(unsafe_code)]

// This crate is intentionally minimal.
// Container implementations are in hs/BHC/Data/*.hs
