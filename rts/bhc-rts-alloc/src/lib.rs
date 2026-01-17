//! Memory allocation primitives for the BHC Runtime System.
//!
//! This crate provides the foundational memory allocation primitives used by
//! the BHC runtime. It defines the core abstractions for memory regions
//! as specified in H26-SPEC Section 9: Memory Model.
//!
//! # Memory Regions
//!
//! BHC defines three allocation regions:
//!
//! 1. **Hot Arena** - Bump allocator, freed at scope end (see `bhc-rts-arena`)
//! 2. **Pinned Heap** - Non-moving memory for FFI/device IO
//! 3. **General Heap** - GC-managed boxed structures (see `bhc-rts-gc`)
//!
//! # Design Goals
//!
//! - Zero-cost abstractions for allocation patterns
//! - Explicit control over memory placement
//! - Safe FFI interop through pinned allocations
//! - Support for SIMD-aligned allocations

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::alloc::{Layout, LayoutError};
use std::ptr::NonNull;

/// Alignment requirements for different allocation purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    /// Default alignment (8 bytes on 64-bit systems).
    Default,
    /// Cache line alignment (64 bytes).
    CacheLine,
    /// SIMD 128-bit alignment (16 bytes).
    Simd128,
    /// SIMD 256-bit alignment (32 bytes, AVX).
    Simd256,
    /// SIMD 512-bit alignment (64 bytes, AVX-512).
    Simd512,
    /// Page alignment (4096 bytes).
    Page,
}

impl Alignment {
    /// Get the alignment value in bytes.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        match self {
            Self::Default => 8,
            Self::CacheLine => 64,
            Self::Simd128 => 16,
            Self::Simd256 => 32,
            Self::Simd512 => 64,
            Self::Page => 4096,
        }
    }
}

/// Memory region classification per H26-SPEC Section 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegion {
    /// Hot arena: bump allocator with scope-based lifetime.
    /// Used for ephemeral allocations in numeric kernels.
    HotArena,
    /// Pinned heap: non-moving memory for FFI and device IO.
    /// Must not be relocated by GC.
    PinnedHeap,
    /// General heap: GC-managed boxed structures.
    /// May be moved during garbage collection.
    GeneralHeap,
    /// GPU device memory: high-bandwidth memory on a GPU device.
    /// Managed separately from host memory.
    DeviceMemory(DeviceMemoryKind),
}

/// Type of GPU device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryKind {
    /// NVIDIA CUDA device memory.
    Cuda(u32),
    /// AMD ROCm/HIP device memory.
    Rocm(u32),
}

/// Metadata for a memory block.
#[derive(Debug, Clone, Copy)]
pub struct BlockMeta {
    /// Size of the allocated block in bytes.
    pub size: usize,
    /// Alignment of the block.
    pub alignment: usize,
    /// Memory region this block belongs to.
    pub region: MemoryRegion,
    /// Whether this block is pinned (cannot be moved by GC).
    pub pinned: bool,
}

/// Result type for allocation operations.
pub type AllocResult<T> = Result<T, AllocError>;

/// Errors that can occur during allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// Out of memory.
    OutOfMemory {
        /// Requested allocation size.
        requested: usize,
    },
    /// Invalid layout (e.g., zero size or invalid alignment).
    InvalidLayout(String),
    /// Arena capacity exhausted.
    ArenaExhausted {
        /// Current arena usage.
        current: usize,
        /// Maximum arena capacity.
        capacity: usize,
    },
    /// Alignment requirement not met.
    AlignmentError {
        /// Requested alignment.
        requested: usize,
        /// Maximum supported alignment.
        supported: usize,
    },
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory { requested } => {
                write!(f, "out of memory: failed to allocate {requested} bytes")
            }
            Self::InvalidLayout(msg) => write!(f, "invalid layout: {msg}"),
            Self::ArenaExhausted { current, capacity } => {
                write!(
                    f,
                    "arena exhausted: {current} bytes used of {capacity} bytes capacity"
                )
            }
            Self::AlignmentError {
                requested,
                supported,
            } => {
                write!(
                    f,
                    "alignment error: requested {requested}, max supported {supported}"
                )
            }
        }
    }
}

impl std::error::Error for AllocError {}

impl From<LayoutError> for AllocError {
    fn from(e: LayoutError) -> Self {
        Self::InvalidLayout(e.to_string())
    }
}

/// Trait for memory allocators in the RTS.
///
/// This trait provides a unified interface for different allocation strategies
/// used by the runtime system.
pub trait Allocator {
    /// Allocate a block of memory with the given layout.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The returned pointer is properly deallocated using `deallocate`
    /// - The memory is not accessed after deallocation
    unsafe fn allocate(&self, layout: Layout) -> AllocResult<NonNull<u8>>;

    /// Deallocate a previously allocated block.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was allocated by this allocator with the same `layout`
    /// - `ptr` has not been deallocated before
    /// - No references to the memory exist after this call
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Allocate zeroed memory.
    ///
    /// # Safety
    ///
    /// Same requirements as `allocate`.
    unsafe fn allocate_zeroed(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        let ptr = unsafe { self.allocate(layout)? };
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, layout.size());
        }
        Ok(ptr)
    }

    /// Reallocate a block of memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was allocated by this allocator with `old_layout`
    /// - `new_layout.size()` is greater than zero
    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_size: usize,
    ) -> AllocResult<NonNull<u8>> {
        let new_layout = Layout::from_size_align(new_size, old_layout.align())?;
        let new_ptr = unsafe { self.allocate(new_layout)? };

        let copy_size = old_layout.size().min(new_size);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), copy_size);
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }
}

/// A raw memory block with associated metadata.
#[derive(Debug)]
pub struct RawBlock {
    ptr: NonNull<u8>,
    layout: Layout,
    region: MemoryRegion,
}

impl RawBlock {
    /// Create a new raw block (for use by allocator implementations).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` points to validly allocated memory
    /// with the given layout.
    #[must_use]
    pub const unsafe fn new(ptr: NonNull<u8>, layout: Layout, region: MemoryRegion) -> Self {
        Self { ptr, layout, region }
    }

    /// Get the pointer to the block's data.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the non-null pointer to the block's data.
    #[inline]
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Get the layout of this block.
    #[inline]
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Get the size of this block in bytes.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> usize {
        self.layout.size()
    }

    /// Get the alignment of this block.
    #[inline]
    #[must_use]
    pub const fn align(&self) -> usize {
        self.layout.align()
    }

    /// Get the memory region this block belongs to.
    #[inline]
    #[must_use]
    pub const fn region(&self) -> MemoryRegion {
        self.region
    }
}

/// Statistics for memory allocation tracking.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocStats {
    /// Total bytes currently allocated.
    pub bytes_allocated: usize,
    /// Total number of allocations performed.
    pub allocation_count: usize,
    /// Total number of deallocations performed.
    pub deallocation_count: usize,
    /// Peak memory usage in bytes.
    pub peak_bytes: usize,
    /// Number of failed allocations.
    pub failed_allocations: usize,
}

impl AllocStats {
    /// Create new empty statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
            peak_bytes: 0,
            failed_allocations: 0,
        }
    }

    /// Record an allocation.
    pub fn record_alloc(&mut self, size: usize) {
        self.bytes_allocated += size;
        self.allocation_count += 1;
        self.peak_bytes = self.peak_bytes.max(self.bytes_allocated);
    }

    /// Record a deallocation.
    pub fn record_dealloc(&mut self, size: usize) {
        self.bytes_allocated = self.bytes_allocated.saturating_sub(size);
        self.deallocation_count += 1;
    }

    /// Record a failed allocation.
    pub fn record_failure(&mut self) {
        self.failed_allocations += 1;
    }
}

/// Utility function to align a size up to the given alignment.
#[inline]
#[must_use]
pub const fn align_up(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (size + align - 1) & !(align - 1)
}

/// Utility function to check if a pointer is aligned.
#[inline]
#[must_use]
pub fn is_aligned(ptr: *const u8, align: usize) -> bool {
    debug_assert!(align.is_power_of_two());
    (ptr as usize) & (align - 1) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_values() {
        assert_eq!(Alignment::Default.as_usize(), 8);
        assert_eq!(Alignment::CacheLine.as_usize(), 64);
        assert_eq!(Alignment::Simd128.as_usize(), 16);
        assert_eq!(Alignment::Simd256.as_usize(), 32);
        assert_eq!(Alignment::Simd512.as_usize(), 64);
        assert_eq!(Alignment::Page.as_usize(), 4096);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(100, 64), 128);
    }

    #[test]
    fn test_is_aligned() {
        let aligned_ptr = 0x1000 as *const u8;
        let unaligned_ptr = 0x1001 as *const u8;

        assert!(is_aligned(aligned_ptr, 8));
        assert!(is_aligned(aligned_ptr, 16));
        assert!(is_aligned(aligned_ptr, 4096));
        assert!(!is_aligned(unaligned_ptr, 8));
    }

    #[test]
    fn test_alloc_stats() {
        let mut stats = AllocStats::new();

        stats.record_alloc(100);
        assert_eq!(stats.bytes_allocated, 100);
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.peak_bytes, 100);

        stats.record_alloc(200);
        assert_eq!(stats.bytes_allocated, 300);
        assert_eq!(stats.peak_bytes, 300);

        stats.record_dealloc(100);
        assert_eq!(stats.bytes_allocated, 200);
        assert_eq!(stats.deallocation_count, 1);
        assert_eq!(stats.peak_bytes, 300); // Peak unchanged
    }

    #[test]
    fn test_alloc_error_display() {
        let err = AllocError::OutOfMemory { requested: 1024 };
        assert!(err.to_string().contains("1024"));

        let err = AllocError::ArenaExhausted {
            current: 500,
            capacity: 1000,
        };
        assert!(err.to_string().contains("500"));
        assert!(err.to_string().contains("1000"));
    }
}
