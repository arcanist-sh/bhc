//! The layout of a constructor object, in one place.
//!
//! Compiled code and the runtime agree on how a data constructor is laid out
//! in memory. The runtime and the standard-library crates used to restate
//! that layout wherever they built or took apart a list, a `Maybe`, a tuple:
//! a dozen private `alloc_cons` and `get_tag`, each with its own offsets.
//! Giving every heap object a uniform header (#94) means changing the layout,
//! which is only possible when it is written down once.
//!
//! # Layout
//!
//! ```text
//! word 0        constructor tag, counted from 0 in declaration order
//! word 1..=n    the fields, one word each
//! ```
//!
//! A field is a pointer to another object or, for some scalar fields, the
//! scalar itself in the word; this module does not distinguish the two.
//!
//! Thunks and closures have other layouts and are not described here yet.
//!
//! # Allocation
//!
//! Objects come from the system allocator, as they did at every site this
//! replaces, and are never freed: there is no collector (#111).

use std::alloc::{alloc, Layout};

/// The size of a word: the tag and each field take one.
pub const WORD: usize = 8;

/// The tag of `[]`.
pub const NIL: i64 = 0;
/// The tag of `(:)`.
pub const CONS: i64 = 1;
/// The tag of `Nothing`.
pub const NOTHING: i64 = 0;
/// The tag of `Just`.
pub const JUST: i64 = 1;
/// The tag of `Left`.
pub const LEFT: i64 = 0;
/// The tag of `Right`.
pub const RIGHT: i64 = 1;
/// The tag of a tuple's only constructor.
pub const TUPLE: i64 = 0;

/// Allocate a constructor object with the given tag and fields.
///
/// # Safety
///
/// The result is uninitialized beyond what is written here, which is all of
/// it; the call is unsafe only because the object's fields are raw pointers
/// the caller vouches for. Aborts nowhere: a failed allocation returns null,
/// which the write then faults on, as the code this replaces did.
#[inline]
#[must_use]
pub unsafe fn alloc_con(tag: i64, fields: &[*mut u8]) -> *mut u8 {
    let size = WORD * (1 + fields.len());
    // SAFETY: `size` is a non-zero multiple of the alignment and far below
    // `isize::MAX` for any field count a constructor can have.
    let object = unsafe { alloc(Layout::from_size_align_unchecked(size, WORD)) };
    // SAFETY: `object` has room for the tag and `fields.len()` words.
    unsafe {
        object.cast::<i64>().write(tag);
        let first = object.cast::<*mut u8>().add(1);
        for (index, field) in fields.iter().enumerate() {
            first.add(index).write(*field);
        }
    }
    object
}

/// The constructor tag of `object`.
///
/// # Safety
///
/// `object` must point to an evaluated constructor object.
#[inline]
#[must_use]
pub unsafe fn tag(object: *const u8) -> i64 {
    // SAFETY: the caller guarantees a constructor object, whose first word
    // is the tag.
    unsafe { object.cast::<i64>().read() }
}

/// Field `index` of `object`, counted from 0.
///
/// # Safety
///
/// `object` must point to an evaluated constructor object with more than
/// `index` fields.
#[inline]
#[must_use]
pub unsafe fn field(object: *const u8, index: usize) -> *mut u8 {
    // SAFETY: the caller guarantees the field exists; fields follow the tag.
    unsafe { object.cast::<*mut u8>().add(1 + index).read() }
}

/// Overwrite field `index` of `object`.
///
/// # Safety
///
/// As for [`field`], and nothing else may be reading the object.
#[inline]
pub unsafe fn set_field(object: *mut u8, index: usize, value: *mut u8) {
    // SAFETY: the caller guarantees the field exists; fields follow the tag.
    unsafe { object.cast::<*mut u8>().add(1 + index).write(value) }
}

/// Allocate `[]`.
///
/// # Safety
///
/// See [`alloc_con`].
#[inline]
#[must_use]
pub unsafe fn nil() -> *mut u8 {
    // SAFETY: no fields to vouch for.
    unsafe { alloc_con(NIL, &[]) }
}

/// Allocate `head : tail`.
///
/// # Safety
///
/// See [`alloc_con`].
#[inline]
#[must_use]
pub unsafe fn cons(head: *mut u8, tail: *mut u8) -> *mut u8 {
    // SAFETY: forwarded to the caller.
    unsafe { alloc_con(CONS, &[head, tail]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_constructor_is_a_tag_followed_by_its_fields() {
        let a = 0x10 as *mut u8;
        let b = 0x20 as *mut u8;
        // SAFETY: the fields are never dereferenced.
        unsafe {
            let object = alloc_con(3, &[a, b]);
            assert_eq!(object as usize % WORD, 0);
            assert_eq!(tag(object), 3);
            assert_eq!(field(object, 0), a);
            assert_eq!(field(object, 1), b);
            // The layout compiled code relies on, spelled out.
            assert_eq!(object.cast::<i64>().read(), 3);
            assert_eq!(object.add(8).cast::<*mut u8>().read(), a);
            assert_eq!(object.add(16).cast::<*mut u8>().read(), b);
            set_field(object, 1, a);
            assert_eq!(field(object, 1), a);
        }
    }

    #[test]
    fn a_list_is_nil_and_cons() {
        let head = 0x30 as *mut u8;
        // SAFETY: the head is never dereferenced.
        unsafe {
            let empty = nil();
            assert_eq!(tag(empty), NIL);
            let list = cons(head, empty);
            assert_eq!(tag(list), CONS);
            assert_eq!(field(list, 0), head);
            assert_eq!(field(list, 1), empty);
        }
    }
}
