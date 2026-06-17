//! Fuzz target for the parser: arbitrary input must produce diagnostics,
//! never a panic or crash.
//!
//! Run with: `cargo +nightly fuzz run parse_module` (requires cargo-fuzz).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(src) = std::str::from_utf8(data) {
        let _ = bhc_parser::parse_module(src, bhc_span::FileId::new(0));
    }
});
