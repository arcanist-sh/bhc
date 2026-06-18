//! Hot arena allocator for the BHC Runtime System.
//!
//! This crate implements the Hot Arena memory region as specified in
//! H26-SPEC Section 9: Memory Model. The hot arena provides:
//!
//! - **Bump allocation** - O(1) allocation via pointer bumping
//! - **Scope-based lifetime** - All allocations freed when arena scope ends
//! - **Zero GC interaction** - Arena memory is invisible to the garbage collector
//!
//! # Usage
//!
//! The hot arena is designed for ephemeral allocations in numeric kernels,
//! where intermediate results have well-defined lifetimes tied to a computation
//! scope.
//!
//! ```ignore
//! use bhc_rts_arena::{HotArena, with_arena};
//!
//! // All allocations are freed when the arena scope ends
//! with_arena(1024 * 1024, |arena| {
//!     let buffer = arena.alloc_slice::<f32>(1000);
//!     compute_kernel(buffer);
//!     // buffer automatically freed here
//! });
//! ```
//!
//! # Design Goals
//!
//! - Minimal allocation overhead for hot paths
//! - No fragmentation within a scope
//! - Deterministic memory usage
//! - Cache-friendly sequential allocation

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod frame;

use bhc_rts_alloc::{
    align_up, Alignment, AllocError, AllocResult, AllocStats, MemoryRegion, Zeroable,
};
use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

/// Default arena size (1 MB).
pub const DEFAULT_ARENA_SIZE: usize = 1024 * 1024;

/// Hot arena allocator with bump allocation strategy.
///
/// The arena allocates memory by simply bumping a pointer forward.
/// All memory is freed at once when the arena is dropped or reset.
///
/// # Invariants
///
/// - `base` points to the start of the allocated region
/// - `cursor` is always >= `base` and <= `base + capacity`
/// - All allocations are properly aligned
#[derive(Debug)]
pub struct HotArena {
    /// Base pointer of the arena memory.
    base: NonNull<u8>,
    /// Current allocation cursor.
    cursor: Cell<*mut u8>,
    /// Total capacity in bytes.
    capacity: usize,
    /// Layout used for the backing allocation; `Drop` must deallocate with
    /// exactly this layout (a mismatched alignment would be UB).
    layout: Layout,
    /// Allocation statistics.
    stats: Cell<AllocStats>,
}

impl HotArena {
    /// Create a new hot arena with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if the system cannot allocate the requested memory.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::with_alignment(capacity, Alignment::CacheLine)
    }

    /// Create a new hot arena with the specified capacity and alignment.
    ///
    /// # Panics
    ///
    /// Panics if the system cannot allocate the requested memory.
    #[must_use]
    pub fn with_alignment(capacity: usize, alignment: Alignment) -> Self {
        let layout =
            Layout::from_size_align(capacity, alignment.as_usize()).expect("invalid arena layout");

        // The global allocator must not be called with a zero-size layout;
        // a capacity-0 arena gets a dangling (aligned) base and never
        // allocates or deallocates.
        let base = if capacity == 0 {
            NonNull::new(alignment.as_usize() as *mut u8).expect("alignment is non-zero")
        } else {
            // Safety: layout is valid and non-zero
            let base = unsafe { alloc(layout) };
            NonNull::new(base).expect("failed to allocate arena memory")
        };

        Self {
            base,
            cursor: Cell::new(base.as_ptr()),
            capacity,
            layout,
            stats: Cell::new(AllocStats::new()),
        }
    }

    /// Create a new hot arena with default capacity (1 MB).
    #[must_use]
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_ARENA_SIZE)
    }

    /// Allocate memory from the arena.
    ///
    /// Returns a pointer to the allocated memory, or an error if the
    /// arena is exhausted.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid until the arena is reset or dropped.
    pub fn alloc_raw(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        let cursor = self.cursor.get();
        let aligned_cursor = align_up(cursor as usize, layout.align());
        let new_cursor = aligned_cursor + layout.size();

        let end = self.base.as_ptr() as usize + self.capacity;
        if new_cursor > end {
            let mut stats = self.stats.get();
            stats.record_failure();
            self.stats.set(stats);

            return Err(AllocError::ArenaExhausted {
                current: self.used(),
                capacity: self.capacity,
            });
        }

        self.cursor.set(new_cursor as *mut u8);

        let mut stats = self.stats.get();
        stats.record_alloc(layout.size());
        self.stats.set(stats);

        // Safety: aligned_cursor is within bounds and properly aligned
        Ok(unsafe { NonNull::new_unchecked(aligned_cursor as *mut u8) })
    }

    /// Allocate a value in the arena.
    ///
    /// The value is moved into the arena and a mutable reference is returned.
    // Returning `&mut` from `&self` is the arena contract: interior bump-pointer
    // mutation hands out exclusive references to fresh, non-overlapping storage.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T>(&self, value: T) -> AllocResult<&mut T> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout)?;

        // Safety: ptr is properly aligned and has enough space for T
        unsafe {
            let ptr = ptr.as_ptr() as *mut T;
            std::ptr::write(ptr, value);
            Ok(&mut *ptr)
        }
    }

    /// Allocate a slice in the arena.
    ///
    /// Returns a mutable slice initialized with the given values.
    #[allow(clippy::mut_from_ref)] // arena contract: see `alloc`
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> AllocResult<&mut [T]> {
        if values.is_empty() {
            return Ok(&mut []);
        }

        let layout = Layout::array::<T>(values.len())
            .map_err(|e| AllocError::InvalidLayout(format!("invalid array layout: {e}")))?;

        let ptr = self.alloc_raw(layout)?;

        // Safety: ptr is properly aligned and has enough space
        unsafe {
            let slice_ptr = ptr.as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(values.as_ptr(), slice_ptr, values.len());
            Ok(std::slice::from_raw_parts_mut(slice_ptr, values.len()))
        }
    }

    /// Allocate an uninitialized slice in the arena.
    ///
    /// The returned slice is `MaybeUninit<T>`, so reading elements before
    /// initializing them is a compile-time obligation rather than UB-by-
    /// convention.
    #[allow(clippy::mut_from_ref)] // arena contract: see `alloc`
    pub fn alloc_slice_uninit<T>(&self, len: usize) -> AllocResult<&mut [MaybeUninit<T>]> {
        if len == 0 {
            return Ok(&mut []);
        }

        let layout = Layout::array::<T>(len)
            .map_err(|e| AllocError::InvalidLayout(format!("invalid array layout: {e}")))?;

        let ptr = self.alloc_raw(layout)?;

        // Safety: ptr is properly aligned and has enough space; MaybeUninit
        // imposes no validity requirement on the contents
        unsafe {
            let slice_ptr = ptr.as_ptr() as *mut MaybeUninit<T>;
            Ok(std::slice::from_raw_parts_mut(slice_ptr, len))
        }
    }

    /// Allocate a zeroed slice in the arena.
    ///
    /// `T: Zeroable` guarantees the all-zero byte pattern is a valid `T`.
    #[allow(clippy::mut_from_ref)] // arena contract: see `alloc`
    pub fn alloc_slice_zeroed<T: Zeroable>(&self, len: usize) -> AllocResult<&mut [T]> {
        if len == 0 {
            return Ok(&mut []);
        }

        let layout = Layout::array::<T>(len)
            .map_err(|e| AllocError::InvalidLayout(format!("invalid array layout: {e}")))?;

        let ptr = self.alloc_raw(layout)?;

        // Safety: ptr is properly aligned and has enough space
        unsafe {
            let slice_ptr = ptr.as_ptr() as *mut T;
            std::ptr::write_bytes(slice_ptr, 0, len);
            Ok(std::slice::from_raw_parts_mut(slice_ptr, len))
        }
    }

    /// Get the total capacity of the arena in bytes.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of bytes currently used.
    #[inline]
    #[must_use]
    pub fn used(&self) -> usize {
        self.cursor.get() as usize - self.base.as_ptr() as usize
    }

    /// Get the number of bytes remaining.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity - self.used()
    }

    /// Check if the arena can accommodate an allocation of the given size.
    #[inline]
    #[must_use]
    pub fn can_alloc(&self, size: usize, align: usize) -> bool {
        let cursor = self.cursor.get() as usize;
        let aligned = align_up(cursor, align);
        aligned + size <= self.base.as_ptr() as usize + self.capacity
    }

    /// Reset the arena, invalidating all previous allocations.
    ///
    /// # Safety
    ///
    /// All references to arena-allocated memory become invalid after this call.
    /// The caller must ensure no such references exist.
    pub unsafe fn reset(&self) {
        self.cursor.set(self.base.as_ptr());
        self.stats.set(AllocStats::new());
    }

    /// Get allocation statistics for this arena.
    #[must_use]
    pub fn stats(&self) -> AllocStats {
        self.stats.get()
    }

    /// Get the memory region type for this arena.
    #[inline]
    #[must_use]
    pub const fn region(&self) -> MemoryRegion {
        MemoryRegion::HotArena
    }
}

impl Drop for HotArena {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }

        // Safety: base was allocated with exactly this layout in
        // `with_alignment`
        unsafe {
            dealloc(self.base.as_ptr(), self.layout);
        }
    }
}

// Safety: HotArena does not provide shared mutable access without Cell
// The Cell is only for internal cursor management
unsafe impl Send for HotArena {}

/// A scope guard that resets the arena when dropped.
///
/// This enables nested arena scopes where inner allocations are
/// freed before outer ones.
///
/// The scope borrows the arena *exclusively*: this is what makes the
/// cursor rewind in `Drop` sound. If the scope held only `&HotArena`,
/// safe code could keep references from `HotArena::alloc` alive across
/// the scope's end and have the rewound cursor reallocate over them.
/// With `&mut`, the borrow checker rules that out, and references handed
/// out by [`ArenaScope::alloc`] are tied to the scope borrow, so they
/// cannot outlive the reset either.
#[derive(Debug)]
pub struct ArenaScope<'a> {
    arena: &'a mut HotArena,
    saved_cursor: *mut u8,
}

impl<'a> ArenaScope<'a> {
    /// Create a new arena scope.
    #[must_use]
    pub fn new(arena: &'a mut HotArena) -> Self {
        let saved_cursor = arena.cursor.get();
        Self {
            arena,
            saved_cursor,
        }
    }

    /// Allocate a value in this scope.
    pub fn alloc<T>(&self, value: T) -> AllocResult<&mut T> {
        self.arena.alloc(value)
    }

    /// Allocate a slice in this scope.
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> AllocResult<&mut [T]> {
        self.arena.alloc_slice(values)
    }

    /// Get bytes used within this scope.
    #[must_use]
    pub fn scope_used(&self) -> usize {
        self.arena.cursor.get() as usize - self.saved_cursor as usize
    }
}

impl Drop for ArenaScope<'_> {
    fn drop(&mut self) {
        self.arena.cursor.set(self.saved_cursor);
    }
}

/// Execute a function with a temporary arena.
///
/// The arena and all its allocations are freed when the function returns.
pub fn with_arena<F, R>(capacity: usize, f: F) -> R
where
    F: FnOnce(&HotArena) -> R,
{
    let arena = HotArena::new(capacity);
    f(&arena)
}

/// Execute a function with a nested arena scope.
///
/// Allocations within the scope are freed when the function returns,
/// but the parent arena remains valid.
pub fn with_scope<'a, F, R>(arena: &'a mut HotArena, f: F) -> R
where
    F: FnOnce(&ArenaScope<'a>) -> R,
{
    let scope = ArenaScope::new(arena);
    f(&scope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic_allocation() {
        let arena = HotArena::new(4096);

        let x = arena.alloc(42i32).unwrap();
        assert_eq!(*x, 42);

        let y = arena.alloc(2.5f64).unwrap();
        assert!((*y - 2.5).abs() < f64::EPSILON);

        assert!(arena.used() > 0);
    }

    #[test]
    fn test_arena_slice_allocation() {
        let arena = HotArena::new(4096);

        let slice = arena.alloc_slice(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        slice[0] = 10;
        assert_eq!(slice[0], 10);
    }

    #[test]
    fn test_arena_zeroed_slice() {
        let arena = HotArena::new(4096);

        let slice: &mut [i32] = arena.alloc_slice_zeroed(100).unwrap();
        assert!(slice.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_arena_exhaustion() {
        let arena = HotArena::new(64);

        // Should succeed
        let _ = arena.alloc_slice::<u8>(&[0; 32]).unwrap();

        // Should fail - not enough space
        let result = arena.alloc_slice::<u8>(&[0; 64]);
        assert!(matches!(result, Err(AllocError::ArenaExhausted { .. })));
    }

    #[test]
    fn test_arena_scope() {
        let mut arena = HotArena::new(4096);

        let _ = arena.alloc(1i32).unwrap();
        let used_before_scope = arena.used();

        {
            let scope = ArenaScope::new(&mut arena);
            let _ = scope.alloc(2i32).unwrap();
            let _ = scope.alloc_slice(&[1, 2, 3]).unwrap();
            assert!(scope.scope_used() > 0);
        }

        // After scope ends, cursor is restored
        assert_eq!(arena.used(), used_before_scope);
    }

    #[test]
    fn test_arena_drop_with_custom_alignment() {
        // Drop must deallocate with the same layout it allocated with
        let arena = HotArena::with_alignment(8192, Alignment::Page);
        let _ = arena.alloc(7u64).unwrap();
        drop(arena);
    }

    #[test]
    fn test_zero_capacity_arena() {
        let arena = HotArena::new(0);
        assert!(arena.alloc(1i32).is_err());
    }

    #[test]
    fn test_alloc_slice_uninit_maybeuninit() {
        let arena = HotArena::new(4096);
        let slice = arena.alloc_slice_uninit::<u32>(4).unwrap();
        for elem in slice.iter_mut() {
            elem.write(7);
        }
    }

    #[test]
    fn test_with_arena() {
        let result = with_arena(1024, |arena| {
            let x = arena.alloc(42).unwrap();
            *x * 2
        });
        assert_eq!(result, 84);
    }

    #[test]
    fn test_arena_stats() {
        let arena = HotArena::new(4096);

        let _ = arena.alloc(42i32).unwrap();
        let _ = arena.alloc(2.5f64).unwrap();

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 2);
        assert!(stats.bytes_allocated > 0);
    }

    #[test]
    fn test_arena_reset() {
        let arena = HotArena::new(4096);

        let _ = arena.alloc(42i32).unwrap();
        assert!(arena.used() > 0);

        unsafe { arena.reset() };
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_empty_slice() {
        let arena = HotArena::new(4096);
        let empty: &mut [i32] = arena.alloc_slice(&[]).unwrap();
        assert!(empty.is_empty());
    }
}
