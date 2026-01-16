//! Generational garbage collector for the BHC Runtime System.
//!
//! This crate implements the garbage collector for the General Heap region
//! as specified in H26-SPEC Section 9: Memory Model. Key features:
//!
//! - **Generational collection** - Young/old generation with different strategies
//! - **Pinned region support** - Objects that must not move (FFI, device IO)
//! - **Write barriers** - Track cross-generation references
//! - **Incremental collection** - Minimize pause times for Server Profile
//!
//! # Architecture
//!
//! The GC manages the General Heap, which contains boxed Haskell values.
//! Objects may be moved during collection unless they are pinned.
//!
//! ```text
//! +------------------+------------------+------------------+
//! |   Nursery (G0)   |   Survivor (G1)  |   Old Gen (G2)   |
//! +------------------+------------------+------------------+
//! |                  |                  |                  |
//! |  Young objects   |  Promoted from   |  Long-lived      |
//! |  Bump alloc      |  G0 after 1      |  objects         |
//! |  Frequent GC     |  survival        |  Rare major GC   |
//! |                  |                  |                  |
//! +------------------+------------------+------------------+
//!
//! +------------------+
//! |   Pinned Region  |
//! +------------------+
//! |                  |
//! |  Non-moving      |
//! |  objects         |
//! |  (FFI, DMA)      |
//! |                  |
//! +------------------+
//! ```
//!
//! # Design Goals
//!
//! - Low latency for Server Profile (bounded pause times)
//! - High throughput for batch processing
//! - Deterministic behavior for Numeric Profile (no GC in hot paths)
//! - Safe FFI interop through pinned allocations

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use bhc_rts_alloc::{AllocError, AllocResult, AllocStats, MemoryRegion};
use parking_lot::{Mutex, RwLock};
use std::alloc::Layout;
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// A wrapper around `NonNull<u8>` that is `Send + Sync`.
///
/// This is used for GC pointers that need to be shared across threads.
/// Safety: The GC ensures these pointers are accessed safely through
/// proper synchronization (write barriers, mutexes, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct GcPtr(NonNull<u8>);

impl GcPtr {
    /// Create a new GcPtr from a NonNull pointer.
    #[inline]
    pub const fn new(ptr: NonNull<u8>) -> Self {
        Self(ptr)
    }

    /// Get the underlying NonNull pointer.
    #[inline]
    pub const fn as_non_null(self) -> NonNull<u8> {
        self.0
    }

    /// Get the raw pointer.
    #[inline]
    pub const fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr()
    }
}

// Safety: GcPtr is used within the GC system with proper synchronization.
// The GC manages these pointers and ensures thread-safe access.
unsafe impl Send for GcPtr {}
unsafe impl Sync for GcPtr {}

/// Generation identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Generation {
    /// Nursery - newly allocated objects.
    Nursery = 0,
    /// Survivor - survived one minor collection.
    Survivor = 1,
    /// Old - long-lived objects.
    Old = 2,
}

impl Generation {
    /// Get the next older generation.
    #[must_use]
    pub const fn promote(self) -> Option<Self> {
        match self {
            Self::Nursery => Some(Self::Survivor),
            Self::Survivor => Some(Self::Old),
            Self::Old => None,
        }
    }
}

/// Object header stored before each GC-managed object.
///
/// The header contains metadata needed for garbage collection:
/// - Mark bits for tracing
/// - Generation information
/// - Forwarding pointer (during collection)
/// - Type information for traversal
#[derive(Debug)]
#[repr(C)]
pub struct ObjectHeader {
    /// Mark bits and flags.
    flags: AtomicU64,
    /// Size of the object in bytes (excluding header).
    size: u32,
    /// Type tag for traversal.
    type_tag: u32,
}

/// Flags stored in the object header.
#[derive(Debug, Clone, Copy)]
pub struct HeaderFlags(u64);

impl HeaderFlags {
    /// Object is marked (reachable).
    pub const MARKED: u64 = 1 << 0;
    /// Object is pinned (cannot be moved).
    pub const PINNED: u64 = 1 << 1;
    /// Object has been forwarded during collection.
    pub const FORWARDED: u64 = 1 << 2;
    /// Object contains pointers.
    pub const HAS_POINTERS: u64 = 1 << 3;

    /// Generation bits (2 bits, positions 4-5).
    const GENERATION_SHIFT: u64 = 4;
    const GENERATION_MASK: u64 = 0b11 << Self::GENERATION_SHIFT;

    /// Create new flags with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Check if the object is marked.
    #[must_use]
    pub const fn is_marked(self) -> bool {
        self.0 & Self::MARKED != 0
    }

    /// Check if the object is pinned.
    #[must_use]
    pub const fn is_pinned(self) -> bool {
        self.0 & Self::PINNED != 0
    }

    /// Check if the object has been forwarded.
    #[must_use]
    pub const fn is_forwarded(self) -> bool {
        self.0 & Self::FORWARDED != 0
    }

    /// Get the generation of this object.
    #[must_use]
    pub const fn generation(self) -> Generation {
        let gen = (self.0 & Self::GENERATION_MASK) >> Self::GENERATION_SHIFT;
        match gen {
            0 => Generation::Nursery,
            1 => Generation::Survivor,
            _ => Generation::Old,
        }
    }

    /// Set the generation.
    #[must_use]
    pub const fn with_generation(self, gen: Generation) -> Self {
        let cleared = self.0 & !Self::GENERATION_MASK;
        Self(cleared | ((gen as u64) << Self::GENERATION_SHIFT))
    }

    /// Set the pinned flag.
    #[must_use]
    pub const fn with_pinned(self, pinned: bool) -> Self {
        if pinned {
            Self(self.0 | Self::PINNED)
        } else {
            Self(self.0 & !Self::PINNED)
        }
    }
}

impl Default for HeaderFlags {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectHeader {
    /// Create a new object header.
    #[must_use]
    pub const fn new(size: u32, type_tag: u32, flags: HeaderFlags) -> Self {
        Self {
            flags: AtomicU64::new(flags.0),
            size,
            type_tag,
        }
    }

    /// Get the size of the object.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size as usize
    }

    /// Get the type tag.
    #[must_use]
    pub const fn type_tag(&self) -> u32 {
        self.type_tag
    }

    /// Get the current flags.
    #[must_use]
    pub fn flags(&self) -> HeaderFlags {
        HeaderFlags(self.flags.load(Ordering::Acquire))
    }

    /// Set the mark bit.
    pub fn mark(&self) {
        self.flags.fetch_or(HeaderFlags::MARKED, Ordering::Release);
    }

    /// Clear the mark bit.
    pub fn unmark(&self) {
        self.flags.fetch_and(!HeaderFlags::MARKED, Ordering::Release);
    }

    /// Check if marked.
    #[must_use]
    pub fn is_marked(&self) -> bool {
        self.flags().is_marked()
    }

    /// Check if pinned.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.flags().is_pinned()
    }
}

/// Configuration for the garbage collector.
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Size of the nursery in bytes.
    pub nursery_size: usize,
    /// Size of the survivor space in bytes.
    pub survivor_size: usize,
    /// Size of the old generation in bytes.
    pub old_gen_size: usize,
    /// Number of collections before promotion from nursery.
    pub nursery_threshold: u32,
    /// Enable incremental collection.
    pub incremental: bool,
    /// Maximum pause time in microseconds (for incremental GC).
    pub max_pause_us: u64,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            nursery_size: 4 * 1024 * 1024,      // 4 MB
            survivor_size: 2 * 1024 * 1024,     // 2 MB
            old_gen_size: 64 * 1024 * 1024,     // 64 MB
            nursery_threshold: 2,
            incremental: false,
            max_pause_us: 1000,
        }
    }
}

/// Statistics from garbage collection.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Number of minor (nursery) collections.
    pub minor_collections: u64,
    /// Number of major (full) collections.
    pub major_collections: u64,
    /// Total bytes collected.
    pub bytes_collected: u64,
    /// Total bytes promoted to older generations.
    pub bytes_promoted: u64,
    /// Total time spent in GC (microseconds).
    pub total_gc_time_us: u64,
    /// Maximum pause time (microseconds).
    pub max_pause_us: u64,
    /// Number of pinned objects.
    pub pinned_objects: u64,
}

/// Handle to a GC-managed object.
///
/// This handle tracks the object's location and can be updated
/// if the object is moved during collection.
#[derive(Debug)]
pub struct GcHandle<T> {
    /// Pointer to the object (may change if object moves).
    ptr: Cell<NonNull<T>>,
    /// Whether this handle is pinned.
    pinned: bool,
}

impl<T> GcHandle<T> {
    /// Create a new GC handle.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid, GC-managed object.
    #[must_use]
    pub const unsafe fn new(ptr: NonNull<T>, pinned: bool) -> Self {
        Self {
            ptr: Cell::new(ptr),
            pinned,
        }
    }

    /// Get the current pointer.
    #[must_use]
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.get().as_ptr()
    }

    /// Get a reference to the object.
    ///
    /// # Safety
    ///
    /// The caller must ensure the object is still alive.
    #[must_use]
    pub unsafe fn as_ref(&self) -> &T {
        unsafe { self.ptr.get().as_ref() }
    }

    /// Check if this handle is pinned.
    #[must_use]
    pub const fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Update the pointer (called by GC after moving).
    ///
    /// # Safety
    ///
    /// Must only be called by the GC during collection.
    pub unsafe fn update_ptr(&self, new_ptr: NonNull<T>) {
        self.ptr.set(new_ptr);
    }
}

/// Root set for garbage collection.
///
/// The root set contains all objects that are directly reachable
/// and should not be collected.
#[derive(Debug, Default)]
pub struct RootSet {
    /// Stack roots (local variables, arguments).
    stack_roots: Vec<GcPtr>,
    /// Global roots (static variables).
    global_roots: Vec<GcPtr>,
    /// Pinned roots (FFI, device IO).
    pinned_roots: Vec<GcPtr>,
}

impl RootSet {
    /// Create a new empty root set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a stack root.
    pub fn add_stack_root(&mut self, ptr: NonNull<u8>) {
        self.stack_roots.push(GcPtr::new(ptr));
    }

    /// Add a global root.
    pub fn add_global_root(&mut self, ptr: NonNull<u8>) {
        self.global_roots.push(GcPtr::new(ptr));
    }

    /// Add a pinned root.
    pub fn add_pinned_root(&mut self, ptr: NonNull<u8>) {
        self.pinned_roots.push(GcPtr::new(ptr));
    }

    /// Clear stack roots (between function calls).
    pub fn clear_stack_roots(&mut self) {
        self.stack_roots.clear();
    }

    /// Iterate over all roots.
    pub fn iter(&self) -> impl Iterator<Item = GcPtr> + '_ {
        self.stack_roots
            .iter()
            .chain(self.global_roots.iter())
            .chain(self.pinned_roots.iter())
            .copied()
    }
}

/// Write barrier for tracking cross-generation references.
///
/// When an old object is mutated to point to a young object,
/// the write barrier records this so the young object is not
/// incorrectly collected.
#[derive(Debug)]
pub struct WriteBarrier {
    /// Remembered set of old->young references.
    remembered_set: Mutex<Vec<GcPtr>>,
    /// Number of barrier invocations.
    invocations: AtomicUsize,
}

impl WriteBarrier {
    /// Create a new write barrier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            remembered_set: Mutex::new(Vec::new()),
            invocations: AtomicUsize::new(0),
        }
    }

    /// Record a write from an old object to a young object.
    pub fn record(&self, old_object: NonNull<u8>) {
        self.invocations.fetch_add(1, Ordering::Relaxed);
        self.remembered_set.lock().push(GcPtr::new(old_object));
    }

    /// Get and clear the remembered set.
    #[must_use]
    pub fn take_remembered_set(&self) -> Vec<GcPtr> {
        std::mem::take(&mut *self.remembered_set.lock())
    }

    /// Get the number of barrier invocations.
    #[must_use]
    pub fn invocations(&self) -> usize {
        self.invocations.load(Ordering::Relaxed)
    }
}

impl Default for WriteBarrier {
    fn default() -> Self {
        Self::new()
    }
}

/// The garbage collector.
///
/// This is the main interface to the GC subsystem.
#[derive(Debug)]
pub struct GarbageCollector {
    /// Configuration.
    config: GcConfig,
    /// Statistics.
    stats: RwLock<GcStats>,
    /// Write barrier.
    write_barrier: WriteBarrier,
    /// Allocation statistics.
    alloc_stats: RwLock<AllocStats>,
    /// Total bytes allocated since last collection.
    bytes_since_gc: AtomicUsize,
}

impl GarbageCollector {
    /// Create a new garbage collector with the given configuration.
    #[must_use]
    pub fn new(config: GcConfig) -> Self {
        Self {
            config,
            stats: RwLock::new(GcStats::default()),
            write_barrier: WriteBarrier::new(),
            alloc_stats: RwLock::new(AllocStats::new()),
            bytes_since_gc: AtomicUsize::new(0),
        }
    }

    /// Create a new garbage collector with default configuration.
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(GcConfig::default())
    }

    /// Allocate memory from the GC heap.
    ///
    /// Objects allocated this way will be managed by the garbage collector.
    pub fn alloc(&self, layout: Layout, pinned: bool) -> AllocResult<NonNull<u8>> {
        let total_size = std::mem::size_of::<ObjectHeader>() + layout.size();

        // Check if we should trigger GC
        let bytes = self.bytes_since_gc.fetch_add(total_size, Ordering::Relaxed);
        if bytes + total_size > self.config.nursery_size {
            // Would trigger GC here in full implementation
            self.bytes_since_gc.store(0, Ordering::Relaxed);
        }

        // For now, use system allocator (full implementation would use generation spaces)
        let header_layout = Layout::new::<ObjectHeader>();
        let (combined_layout, offset) = header_layout
            .extend(layout)
            .map_err(|_| AllocError::InvalidLayout("layout overflow".into()))?;

        let ptr = unsafe { std::alloc::alloc(combined_layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocError::OutOfMemory {
            requested: combined_layout.size(),
        })?;

        // Initialize header
        let flags = HeaderFlags::new()
            .with_generation(Generation::Nursery)
            .with_pinned(pinned);
        let header = ObjectHeader::new(layout.size() as u32, 0, flags);
        unsafe {
            std::ptr::write(ptr.as_ptr() as *mut ObjectHeader, header);
        }

        // Update stats
        {
            let mut stats = self.alloc_stats.write();
            stats.record_alloc(total_size);
        }

        // Return pointer to data (after header)
        let data_ptr = unsafe { ptr.as_ptr().add(offset) };
        Ok(unsafe { NonNull::new_unchecked(data_ptr) })
    }

    /// Allocate a pinned object that will not be moved by GC.
    pub fn alloc_pinned(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        self.alloc(layout, true)
    }

    /// Trigger a minor (nursery) collection.
    pub fn minor_collect(&self, _roots: &RootSet) {
        // Placeholder: Full implementation would:
        // 1. Mark all reachable objects from roots
        // 2. Copy live objects to survivor space
        // 3. Update remembered set
        // 4. Free nursery

        let mut stats = self.stats.write();
        stats.minor_collections += 1;
    }

    /// Trigger a major (full) collection.
    pub fn major_collect(&self, _roots: &RootSet) {
        // Placeholder: Full implementation would:
        // 1. Mark all reachable objects from roots
        // 2. Sweep/compact all generations
        // 3. Free unreachable objects

        let mut stats = self.stats.write();
        stats.major_collections += 1;
    }

    /// Get the write barrier for recording mutations.
    #[must_use]
    pub fn write_barrier(&self) -> &WriteBarrier {
        &self.write_barrier
    }

    /// Get GC statistics.
    #[must_use]
    pub fn stats(&self) -> GcStats {
        self.stats.read().clone()
    }

    /// Get the memory region managed by this GC.
    #[must_use]
    pub const fn region(&self) -> MemoryRegion {
        MemoryRegion::GeneralHeap
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &GcConfig {
        &self.config
    }
}

impl Default for GarbageCollector {
    fn default() -> Self {
        Self::with_default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_promotion() {
        assert_eq!(Generation::Nursery.promote(), Some(Generation::Survivor));
        assert_eq!(Generation::Survivor.promote(), Some(Generation::Old));
        assert_eq!(Generation::Old.promote(), None);
    }

    #[test]
    fn test_header_flags() {
        let flags = HeaderFlags::new();
        assert!(!flags.is_marked());
        assert!(!flags.is_pinned());
        assert_eq!(flags.generation(), Generation::Nursery);

        let flags = flags.with_generation(Generation::Old).with_pinned(true);
        assert!(flags.is_pinned());
        assert_eq!(flags.generation(), Generation::Old);
    }

    #[test]
    fn test_object_header() {
        let flags = HeaderFlags::new().with_generation(Generation::Survivor);
        let header = ObjectHeader::new(64, 42, flags);

        assert_eq!(header.size(), 64);
        assert_eq!(header.type_tag(), 42);
        assert!(!header.is_marked());

        header.mark();
        assert!(header.is_marked());

        header.unmark();
        assert!(!header.is_marked());
    }

    #[test]
    fn test_gc_alloc() {
        let gc = GarbageCollector::with_default_config();

        let layout = Layout::new::<[u64; 10]>();
        let ptr = gc.alloc(layout, false).unwrap();

        // Verify we got a valid pointer
        assert!(!ptr.as_ptr().is_null());
    }

    #[test]
    fn test_gc_pinned_alloc() {
        let gc = GarbageCollector::with_default_config();

        let layout = Layout::new::<[u64; 10]>();
        let ptr = gc.alloc_pinned(layout).unwrap();

        assert!(!ptr.as_ptr().is_null());
    }

    #[test]
    fn test_write_barrier() {
        let barrier = WriteBarrier::new();

        let ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        barrier.record(ptr);

        assert_eq!(barrier.invocations(), 1);

        let remembered = barrier.take_remembered_set();
        assert_eq!(remembered.len(), 1);
        assert_eq!(remembered[0], GcPtr::new(ptr));

        // After take, should be empty
        let remembered = barrier.take_remembered_set();
        assert!(remembered.is_empty());
    }

    #[test]
    fn test_root_set() {
        let mut roots = RootSet::new();

        let stack_ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        let global_ptr = NonNull::new(0x2000 as *mut u8).unwrap();
        let pinned_ptr = NonNull::new(0x3000 as *mut u8).unwrap();

        roots.add_stack_root(stack_ptr);
        roots.add_global_root(global_ptr);
        roots.add_pinned_root(pinned_ptr);

        let all_roots: Vec<_> = roots.iter().collect();
        assert_eq!(all_roots.len(), 3);

        roots.clear_stack_roots();
        let all_roots: Vec<_> = roots.iter().collect();
        assert_eq!(all_roots.len(), 2);
    }

    #[test]
    fn test_gc_stats() {
        let gc = GarbageCollector::with_default_config();
        let roots = RootSet::new();

        gc.minor_collect(&roots);
        gc.minor_collect(&roots);
        gc.major_collect(&roots);

        let stats = gc.stats();
        assert_eq!(stats.minor_collections, 2);
        assert_eq!(stats.major_collections, 1);
    }
}
