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
/// library), so this mirrors its thunk protocol over the layout in
/// `bhc_rts_alloc::object`: a thunk is claimed by CAS to a blackhole,
/// evaluated, and published as an indirection; any other first word is the
/// tag of an evaluated value.
///
/// # Safety
///
/// `obj` must be null or a valid BHC heap object.
#[inline]
pub unsafe fn force_heap_object(obj: *mut u8) -> *mut u8 {
    use bhc_rts_alloc::object::{self, BLACKHOLE, INDIRECTION, THUNK};
    use std::sync::atomic::Ordering;
    if obj.is_null() {
        return obj;
    }
    // SAFETY: the caller guarantees a valid heap object; a thunk state in
    // its first word means it has a thunk's layout.
    unsafe {
        let state = object::state(obj);
        match state.load(Ordering::Acquire) {
            INDIRECTION => object::thunk_slot(obj).load(Ordering::Acquire),
            THUNK => {
                if state
                    .compare_exchange(THUNK, BLACKHOLE, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    panic!("<<loop>> - circular reference in thunk evaluation");
                }
                let slot = object::thunk_slot(obj);
                let eval_fn: extern "C" fn(*mut u8) -> *mut u8 =
                    std::mem::transmute(slot.load(Ordering::Relaxed));
                let result = eval_fn(object::thunk_env(obj));
                slot.store(result, Ordering::Release);
                state.store(INDIRECTION, Ordering::Release);
                result
            }
            BLACKHOLE => panic!("<<loop>> - circular reference in thunk evaluation"),
            _ => obj,
        }
    }
}
