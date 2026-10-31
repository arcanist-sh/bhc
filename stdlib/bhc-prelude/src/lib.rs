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

/// Evaluate a heap object to weak head normal form, as the RTS's `bhc_force`
/// does, for stdlib code that walks Haskell lists.
///
/// A cons tail may be an unevaluated thunk: a corecursive producer, or the
/// lazy rest of a list builtin that stopped at one (#54). The stdlib crates
/// do not link `bhc-rts` (its symbols would be duplicated in each static
/// library), so this mirrors its thunk protocol: tag `-1` thunk (claimed by
/// CAS to the `-2` blackhole, evaluated, published as a `-3` indirection),
/// tag `-3` indirection, tag `>= 0` an evaluated value.
///
/// # Safety
///
/// `obj` must be null or a valid BHC heap object.
#[inline]
pub unsafe fn force_heap_object(obj: *mut u8) -> *mut u8 {
    use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};
    const THUNK: i64 = -1;
    const BLACKHOLE: i64 = -2;
    const INDIRECTION: i64 = -3;
    if obj.is_null() {
        return obj;
    }
    // SAFETY: the caller guarantees a valid heap object; word 0 is its tag
    // and, for a thunk or indirection, word 1 is its eval fn or result.
    unsafe {
        let tag = &*(obj as *const AtomicI64);
        match tag.load(Ordering::Acquire) {
            INDIRECTION => (*(obj.add(8) as *const AtomicPtr<u8>)).load(Ordering::Acquire),
            THUNK => {
                if tag
                    .compare_exchange(THUNK, BLACKHOLE, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    panic!("<<loop>> - circular reference in thunk evaluation");
                }
                let eval_fn: extern "C" fn(*mut u8) -> *mut u8 =
                    std::mem::transmute(*(obj.add(8) as *const *const u8));
                let result = eval_fn(obj.add(24));
                (*(obj.add(8) as *const AtomicPtr<u8>)).store(result, Ordering::Release);
                tag.store(INDIRECTION, Ordering::Release);
                result
            }
            BLACKHOLE => panic!("<<loop>> - circular reference in thunk evaluation"),
            _ => obj,
        }
    }
}
