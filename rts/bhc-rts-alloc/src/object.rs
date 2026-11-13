//! The layout of a constructor object, in one place.
//!
//! Compiled code and the runtime agree on how a data constructor is laid out
//! in memory. The runtime and the standard-library crates used to restate
//! that layout wherever they built or took apart a list, a `Maybe`, a tuple:
//! a dozen private `alloc_cons` and `get_tag`, each with its own offsets.
//! Giving every heap object a uniform header (#94) means changing the layout,
//! which is only possible when it is written down once.
//!
//! # Constructors
//!
//! ```text
//! word 0        constructor tag, counted from 0 in declaration order
//! word 1..=n    the fields, one word each
//! ```
//!
//! A field is a pointer to another object or, for some scalar fields, the
//! scalar itself in the word; this module does not distinguish the two.
//!
//! # Thunks
//!
//! Word 0 says what an object is. A constructor tag is never negative; a
//! negative value there is a thunk state, [`CLOSURE`], or the kind of an
//! object the runtime owns.
//!
//! ```text
//! word 0        state: THUNK, BLACKHOLE or INDIRECTION
//! word 1        THUNK, BLACKHOLE: the code, `fn(env) -> value`
//!               INDIRECTION: the value
//! word 2        THUNK: the number of captured values (nothing reads it)
//!               BLACKHOLE: the thread evaluating it
//! word 3..      the captured values, which the code receives a pointer to
//! ```
//!
//! Words 0 and 1 are read and written atomically, since a thunk can be
//! shared between threads.
//!
//! # Closures
//!
//! ```text
//! word 0        CLOSURE
//! word 1        the code, `fn(closure, args...) -> value`
//! word 2        the number of arguments the code takes
//! word 3..      the captured values
//! ```
//!
//! The code and the captured values sit where a thunk's do.
//!
//! A partial application is a closure whose captured values are the closure
//! it applies, the number of arguments held so far, and those arguments.
//!
//! # Objects the runtime owns
//!
//! An `Integer`, an `IORef`, an `MVar`, a `Chan` and a `ThreadId` are Rust
//! values behind a kind word:
//!
//! ```text
//! word 0        INTEGER, IOREF, MVAR, CHAN or THREAD_ID
//! word 1..      the runtime's own data, not described here
//! ```
//!
//! A strict `Text` and a strict `ByteString` share one layout, and the text
//! library builds either from the other's bytes, so they share a kind:
//!
//! ```text
//! word 0        BYTES
//! word 1        pointer to a byte buffer, which slices share
//! word 2        offset of the first byte in the buffer
//! word 3        length in bytes
//! ```
//!
//! A value that owns its bytes keeps them after word 3 and points word 1
//! there.
//!
//! A `Map` (which `IntMap` shares) and a `Set` are Rust collections behind
//! [`MAP`] and [`SET`], like the objects above.
//!
//! # Allocation
//!
//! Constructors built here come from the system allocator, as they did at
//! every site this replaces, and are never freed: there is no collector
//! (#111). Thunks and closures are initialized in memory the caller
//! allocates.

use std::alloc::{alloc, Layout};
use std::sync::atomic::{AtomicI64, AtomicPtr};

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

/// Word 0 of a thunk that has not been evaluated.
pub const THUNK: i64 = -1;
/// Word 0 of a thunk that is being evaluated.
pub const BLACKHOLE: i64 = -2;
/// Word 0 of a thunk that has been evaluated; word 1 is its value.
pub const INDIRECTION: i64 = -3;

/// The size in bytes of a thunk that captures `captured` values.
#[inline]
#[must_use]
pub const fn thunk_size(captured: usize) -> usize {
    WORD * (3 + captured)
}

/// Make `object` an unevaluated thunk running `code` over `captured`.
///
/// # Safety
///
/// `object` must be word-aligned, writable for [`thunk_size`] of
/// `captured.len()` bytes, and not yet visible to another thread. `code`
/// must be an `extern "C" fn(*mut u8) -> *mut u8` taking the captured values.
#[inline]
pub unsafe fn init_thunk(object: *mut u8, code: *const u8, captured: &[*mut u8]) {
    // SAFETY: the caller guarantees room for three words and the captures.
    unsafe {
        object.cast::<i64>().write(THUNK);
        object.cast::<*const u8>().add(1).write(code);
        object.cast::<usize>().add(2).write(captured.len());
        let first = object.cast::<*mut u8>().add(3);
        for (index, value) in captured.iter().enumerate() {
            first.add(index).write(*value);
        }
    }
}

/// The state word of a thunk, which is also the tag word of a constructor.
///
/// # Safety
///
/// `object` must point to a heap object that outlives the reference.
#[inline]
#[must_use]
pub unsafe fn state<'a>(object: *mut u8) -> &'a AtomicI64 {
    // SAFETY: the caller guarantees a heap object, whose first word this is.
    unsafe { &*object.cast::<AtomicI64>() }
}

/// Word 1 of a thunk: its code until it is evaluated, then its value.
///
/// # Safety
///
/// `object` must point to a thunk that outlives the reference.
#[inline]
#[must_use]
pub unsafe fn thunk_slot<'a>(object: *mut u8) -> &'a AtomicPtr<u8> {
    // SAFETY: the caller guarantees a thunk, which has a second word.
    unsafe { &*object.cast::<AtomicPtr<u8>>().add(1) }
}

/// The thread evaluating a blackholed thunk.
///
/// # Safety
///
/// `object` must point to a thunk.
#[inline]
#[must_use]
pub unsafe fn thunk_owner(object: *const u8) -> u64 {
    // SAFETY: the caller guarantees a thunk, which has a third word.
    unsafe { object.cast::<u64>().add(2).read() }
}

/// Record the thread evaluating a thunk.
///
/// # Safety
///
/// `object` must point to a thunk the caller has blackholed.
#[inline]
pub unsafe fn set_thunk_owner(object: *mut u8, thread: u64) {
    // SAFETY: the caller guarantees a thunk, which has a third word.
    unsafe { object.cast::<u64>().add(2).write(thread) }
}

/// The captured values of a thunk, as its code receives them.
///
/// # Safety
///
/// `object` must point to a thunk.
#[inline]
#[must_use]
pub unsafe fn thunk_env(object: *mut u8) -> *mut u8 {
    // SAFETY: the captures follow the three header words.
    unsafe { object.add(3 * WORD) }
}

/// Word 0 of a closure.
pub const CLOSURE: i64 = -4;

/// Word 0 of an `Integer`.
pub const INTEGER: i64 = -16;
/// Word 0 of an `IORef`.
pub const IOREF: i64 = -17;
/// Word 0 of an `MVar`.
pub const MVAR: i64 = -18;
/// Word 0 of a `Chan`.
pub const CHAN: i64 = -19;
/// Word 0 of a `ThreadId`.
pub const THREAD_ID: i64 = -20;

/// The number of words in a closure before its captured values.
pub const CLOSURE_HEADER_WORDS: usize = 3;

/// The size in bytes of a closure that captures `captured` values.
#[inline]
#[must_use]
pub const fn closure_size(captured: usize) -> usize {
    WORD * (CLOSURE_HEADER_WORDS + captured)
}

/// Whether `object` is a closure.
///
/// # Safety
///
/// `object` must point to a heap object.
#[inline]
#[must_use]
pub unsafe fn is_closure(object: *const u8) -> bool {
    // SAFETY: every heap object has a first word.
    unsafe { tag(object) == CLOSURE }
}

/// Make `object` a closure over `captured` whose `code` takes `arity`
/// arguments.
///
/// # Safety
///
/// `object` must be word-aligned and writable for [`closure_size`] of
/// `captured.len()` bytes. `code` must be an `extern "C"` function taking the
/// closure and then `arity` arguments.
#[inline]
pub unsafe fn init_closure(object: *mut u8, code: *const u8, arity: usize, captured: &[*mut u8]) {
    // SAFETY: the caller guarantees room for the header and the captures.
    unsafe {
        object.cast::<i64>().write(CLOSURE);
        object.cast::<*const u8>().add(1).write(code);
        object.cast::<usize>().add(2).write(arity);
        let first = object.cast::<*mut u8>().add(CLOSURE_HEADER_WORDS);
        for (index, value) in captured.iter().enumerate() {
            first.add(index).write(*value);
        }
    }
}

/// The code of a closure.
///
/// # Safety
///
/// `object` must point to a closure.
#[inline]
#[must_use]
pub unsafe fn closure_code(object: *const u8) -> *const u8 {
    // SAFETY: the caller guarantees a closure, whose second word is its code.
    unsafe { object.cast::<*const u8>().add(1).read() }
}

/// The number of arguments a closure's code takes.
///
/// # Safety
///
/// `object` must point to a closure.
#[inline]
#[must_use]
pub unsafe fn closure_arity(object: *const u8) -> usize {
    // SAFETY: the caller guarantees a closure, which has a third word.
    unsafe { object.cast::<usize>().add(2).read() }
}

/// Captured value `index` of a closure, counted from 0.
///
/// # Safety
///
/// `object` must point to a closure with more than `index` captured values.
#[inline]
#[must_use]
pub unsafe fn closure_captured(object: *const u8, index: usize) -> *mut u8 {
    // SAFETY: the caller guarantees the value exists; captures follow the
    // header.
    unsafe {
        object
            .cast::<*mut u8>()
            .add(CLOSURE_HEADER_WORDS + index)
            .read()
    }
}

/// Overwrite captured value `index` of a closure.
///
/// # Safety
///
/// As for [`closure_captured`], and nothing else may be reading the closure.
#[inline]
pub unsafe fn set_closure_captured(object: *mut u8, index: usize, value: *mut u8) {
    // SAFETY: the caller guarantees the value exists; captures follow the
    // header.
    unsafe {
        object
            .cast::<*mut u8>()
            .add(CLOSURE_HEADER_WORDS + index)
            .write(value)
    }
}

/// Word 0 of a strict `Text` or `ByteString`.
pub const BYTES: i64 = -21;
/// Word 0 of a `Map` or an `IntMap`.
pub const MAP: i64 = -22;
/// Word 0 of a `Set` or an `IntSet`.
pub const SET: i64 = -23;

/// The size in bytes of a `Text` or `ByteString` before any bytes it owns.
pub const BYTES_HEADER: usize = 4 * WORD;

/// Make `object` a `Text` or `ByteString` of `len` bytes starting `offset`
/// bytes into `buffer`.
///
/// # Safety
///
/// `object` must be word-aligned and writable for [`BYTES_HEADER`] bytes.
#[inline]
pub unsafe fn init_bytes(object: *mut u8, buffer: *const u8, offset: usize, len: usize) {
    // SAFETY: the caller guarantees room for the four header words.
    unsafe {
        object.cast::<i64>().write(BYTES);
        object.cast::<*const u8>().add(1).write(buffer);
        object.cast::<usize>().add(2).write(offset);
        object.cast::<usize>().add(3).write(len);
    }
}

/// The byte buffer of a `Text` or `ByteString`.
///
/// # Safety
///
/// `object` must point to a `Text` or `ByteString`.
#[inline]
#[must_use]
pub unsafe fn bytes_buffer(object: *const u8) -> *const u8 {
    // SAFETY: the caller guarantees the layout above.
    unsafe { object.cast::<*const u8>().add(1).read() }
}

/// The offset of a `Text` or `ByteString`'s first byte in its buffer.
///
/// # Safety
///
/// `object` must point to a `Text` or `ByteString`.
#[inline]
#[must_use]
pub unsafe fn bytes_offset(object: *const u8) -> usize {
    // SAFETY: the caller guarantees the layout above.
    unsafe { object.cast::<usize>().add(2).read() }
}

/// The length in bytes of a `Text` or `ByteString`.
///
/// # Safety
///
/// `object` must point to a `Text` or `ByteString`.
#[inline]
#[must_use]
pub unsafe fn bytes_len(object: *const u8) -> usize {
    // SAFETY: the caller guarantees the layout above.
    unsafe { object.cast::<usize>().add(3).read() }
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
    fn a_thunk_is_state_code_size_and_captures() {
        use std::sync::atomic::Ordering;
        let code = 0x40 as *const u8;
        let a = 0x10 as *mut u8;
        let b = 0x20 as *mut u8;
        let mut words = [0usize; 5];
        let object = words.as_mut_ptr().cast::<u8>();
        assert_eq!(thunk_size(2), 40);
        // SAFETY: `words` has room for a thunk with two captures.
        unsafe {
            init_thunk(object, code, &[a, b]);
            assert_eq!(state(object).load(Ordering::Relaxed), THUNK);
            assert_eq!(thunk_slot(object).load(Ordering::Relaxed), code.cast_mut());
            assert_eq!(thunk_env(object), object.add(24));
            set_thunk_owner(object, 7);
            assert_eq!(thunk_owner(object), 7);
        }
        // The layout compiled code relies on, spelled out.
        assert_eq!(words, [THUNK as usize, 0x40, 7, 0x10, 0x20]);
    }

    #[test]
    fn a_closure_is_kind_code_arity_and_captures() {
        let code = 0x40 as *const u8;
        let a = 0x10 as *mut u8;
        let mut words = [0usize; 4];
        let object = words.as_mut_ptr().cast::<u8>();
        assert_eq!(closure_size(1), 32);
        // SAFETY: `words` has room for a closure with one capture.
        unsafe {
            init_closure(object, code, 2, &[a]);
            assert_eq!(closure_code(object), code);
            assert_eq!(closure_arity(object), 2);
            assert_eq!(closure_captured(object, 0), a);
            assert!(is_closure(object));
        }
        // The layout compiled code relies on, spelled out.
        assert_eq!(words, [CLOSURE as usize, 0x40, 2, 0x10]);
    }

    #[test]
    fn text_is_kind_buffer_offset_and_length() {
        let buffer = 0x50 as *const u8;
        let mut words = [0usize; 4];
        let object = words.as_mut_ptr().cast::<u8>();
        assert_eq!(BYTES_HEADER, 32);
        // SAFETY: `words` has room for the header; the buffer is never read.
        unsafe {
            init_bytes(object, buffer, 3, 9);
            assert_eq!(tag(object), BYTES);
            assert_eq!(bytes_buffer(object), buffer);
            assert_eq!(bytes_offset(object), 3);
            assert_eq!(bytes_len(object), 9);
        }
        assert_eq!(words, [BYTES as usize, 0x50, 3, 9]);
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
