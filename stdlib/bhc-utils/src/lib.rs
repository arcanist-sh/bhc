//! BHC Utils Library - Rust support
//!
//! Utility primitives for BHC.
//!
//! # Architecture
//!
//! The high-level utility APIs are **defined in Haskell** (see
//! `hs/BHC/Data/Time.hs`, `hs/BHC/Data/Random.hs`, `hs/BHC/Data/JSON.hs`).
//! This Rust crate provides FFI primitives for operations requiring
//! system access or high-performance implementations.
//!
//! # What belongs here
//!
//! - Time: System clock access (`now`, `elapsed`)
//! - Random: Secure seeding, high-performance RNG state
//! - JSON: Performance-critical parsing (optional, could be pure Haskell)
//!
//! # What does NOT belong here
//!
//! - Duration/Date/Time types (those could be Haskell)
//! - JSON type definitions (those should be Haskell)
//! - High-level random distributions (those are Haskell)
//!
//! # FFI Exports
//!
//! - Time: `bhc_time_now`, `bhc_time_elapsed`
//! - Random: `bhc_rng_new`, `bhc_rng_next_u64`, `bhc_rng_seed`
//! - JSON: `bhc_json_parse`, `bhc_json_serialize`
//!
//! # Modules
//!
//! - [`time`] - Date, time, and duration operations
//! - [`random`] - Random number generation
//! - [`json`] - JSON parsing and serialization

#![warn(missing_docs)]
#![warn(unsafe_code)]

pub mod json;
pub mod random;
pub mod time;

// Re-export main types
pub use json::{Json, JsonError, JsonResult};
pub use random::Rng;
pub use time::{Date, DateTime, Duration, Instant, Time};
