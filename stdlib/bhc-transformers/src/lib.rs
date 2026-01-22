//! BHC Transformers Library - Rust support
//!
//! This crate provides minimal Rust support for BHC transformers.
//!
//! # Architecture
//!
//! Monad transformers (ReaderT, StateT, WriterT, ExceptT, etc.) are
//! **implemented in Haskell** (see `hs/BHC/Control/Monad/*.hs`).
//! BHC compiles them directly.
//!
//! Monad transformers are quintessentially Haskell - they don't need
//! any Rust implementation. This crate is intentionally empty.

#![warn(missing_docs)]
#![allow(unsafe_code)]

// This crate is intentionally minimal.
// Transformer implementations are in hs/BHC/Control/Monad/*.hs
