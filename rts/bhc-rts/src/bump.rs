//! The allocator behind `bhc_alloc`: bump allocation from thread-local
//! chunks.
//!
//! Compiled code allocates a heap object for every constructor, closure and
//! thunk, and nothing frees them: there is no collector yet (#111). Each
//! object used to be its own `malloc`, which costs a call into the system
//! allocator and its per-block bookkeeping for objects that are mostly 16 to
//! 48 bytes.
//!
//! Here an allocation is a pointer increment within a chunk the thread owns.
//! That is also the shape a nursery has, which is what the collector will
//! need (#94, #112). A request too large to be worth a chunk's space goes to
//! the system allocator as before.
//!
//! Chunks are never returned: an object may be reachable from any thread,
//! and nothing knows when it is not.

use std::alloc::{alloc, Layout};
use std::cell::Cell;
use std::sync::Mutex;

/// The size of a chunk.
const CHUNK_SIZE: usize = 1 << 20;

/// Requests above this go to the system allocator: a chunk should hold many
/// objects, not be mostly one.
const LARGE_OBJECT: usize = CHUNK_SIZE / 16;

/// Every object is aligned to this, as `bhc_alloc` always has been.
const ALIGN: usize = 8;

thread_local! {
    /// The next free address in this thread's chunk, and the chunk's end.
    /// Both zero before the first allocation.
    static CURSOR: Cell<usize> = const { Cell::new(0) };
    static END: Cell<usize> = const { Cell::new(0) };
}

/// The address range of every chunk handed out, so `bhc_free` can tell a
/// bump-allocated pointer (which it must leave alone) from one the system
/// allocator owns. Consulted only by [`owns`], which nothing calls on a hot
/// path.
static CHUNKS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Start a new chunk for this thread. `None` when the system has no memory.
fn new_chunk() -> Option<(usize, usize)> {
    let layout = Layout::from_size_align(CHUNK_SIZE, 16).ok()?;
    // SAFETY: the layout has a non-zero size.
    let start = unsafe { alloc(layout) } as usize;
    if start == 0 {
        return None;
    }
    let end = start + CHUNK_SIZE;
    match CHUNKS.lock() {
        Ok(mut chunks) => chunks.push((start, end)),
        Err(poisoned) => poisoned.into_inner().push((start, end)),
    }
    Some((start, end))
}

/// Allocate `size` bytes, aligned to 8. Null when the system has no memory.
/// `size` must not be zero.
pub(crate) fn allocate(size: usize) -> *mut u8 {
    let Some(rounded) = size.checked_add(ALIGN - 1).map(|n| n & !(ALIGN - 1)) else {
        return std::ptr::null_mut();
    };
    if rounded > LARGE_OBJECT {
        return match Layout::from_size_align(size, ALIGN) {
            // SAFETY: `size` is non-zero, so the layout is.
            Ok(layout) => unsafe { alloc(layout) },
            Err(_) => std::ptr::null_mut(),
        };
    }
    CURSOR.with(|cursor| {
        END.with(|end| {
            let mut at = cursor.get();
            if at == 0 || at + rounded > end.get() {
                let Some((start, chunk_end)) = new_chunk() else {
                    return std::ptr::null_mut();
                };
                at = start;
                end.set(chunk_end);
            }
            cursor.set(at + rounded);
            at as *mut u8
        })
    })
}

/// Whether `ptr` points into a chunk, i.e. came from [`allocate`]'s bump
/// path and must not be handed to the system allocator.
pub(crate) fn owns(ptr: *const u8) -> bool {
    let address = ptr as usize;
    let chunks = match CHUNKS.lock() {
        Ok(chunks) => chunks,
        Err(poisoned) => poisoned.into_inner(),
    };
    chunks
        .iter()
        .any(|(start, end)| (*start..*end).contains(&address))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_objects_are_aligned_distinct_and_writable() {
        let mut seen = Vec::new();
        for size in [1usize, 7, 8, 9, 16, 24, 40, 1000] {
            let p = allocate(size);
            assert!(!p.is_null());
            assert_eq!(p as usize % ALIGN, 0, "size {size}");
            assert!(owns(p));
            // SAFETY: `p` is a fresh allocation of at least `size` bytes.
            unsafe { std::ptr::write_bytes(p, 0xAB, size) };
            seen.push((p as usize, size));
        }
        seen.sort_unstable();
        for pair in seen.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0, "objects overlap");
        }
    }

    #[test]
    fn consecutive_objects_are_adjacent() {
        // Within a chunk an allocation is a pointer increment.
        let first = allocate(16) as usize;
        let second = allocate(16) as usize;
        let third = allocate(5) as usize;
        let fourth = allocate(8) as usize;
        if owns((first + 64) as *const u8) {
            assert_eq!(second, first + 16);
            assert_eq!(third, second + 16);
            assert_eq!(fourth, third + 8);
        }
    }

    #[test]
    fn a_chunk_is_replaced_when_it_is_full() {
        let object = LARGE_OBJECT;
        let count = CHUNK_SIZE / object + 2;
        let pointers: Vec<usize> = (0..count).map(|_| allocate(object) as usize).collect();
        assert!(pointers.iter().all(|p| *p != 0 && owns(*p as *const u8)));
        let mut sorted = pointers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), count);
    }

    #[test]
    fn large_objects_use_the_system_allocator() {
        let size = LARGE_OBJECT + 8;
        let p = allocate(size);
        assert!(!p.is_null());
        assert!(!owns(p));
        // SAFETY: allocated just above with this layout.
        unsafe { std::alloc::dealloc(p, Layout::from_size_align(size, ALIGN).unwrap()) };
    }

    #[test]
    fn each_thread_has_its_own_chunk() {
        let here = allocate(16) as usize;
        let there = std::thread::spawn(|| allocate(16) as usize).join().unwrap();
        assert_ne!(here, there);
        assert!(
            owns(there as *const u8),
            "the other thread's chunk outlives it"
        );
    }
}
