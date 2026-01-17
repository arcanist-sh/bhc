//! GPU runtime abstraction layer.
//!
//! This module provides a runtime abstraction that unifies CUDA and ROCm
//! APIs behind a common interface.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                    GpuContext                        │
//! ├─────────────────────────────────────────────────────┤
//! │                GpuRuntime trait                      │
//! ├───────────────────────┬─────────────────────────────┤
//! │   CudaRuntime         │       RocmRuntime           │
//! │   (cuda feature)      │       (rocm feature)        │
//! ├───────────────────────┼─────────────────────────────┤
//! │   CUDA Driver API     │       HIP API               │
//! │   (cuCtxCreate, etc.) │       (hipCtxCreate, etc.)  │
//! └───────────────────────┴─────────────────────────────┘
//! ```
//!
//! # Runtime Selection
//!
//! The appropriate runtime is selected based on:
//! 1. Compile-time features (`cuda`, `rocm`)
//! 2. Device type detected at runtime

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod rocm;

use crate::device::{DeviceId, DeviceInfo, DeviceKind};
use crate::memory::DevicePtr;
use crate::GpuResult;

/// Trait for GPU runtime implementations.
///
/// This trait abstracts over CUDA and ROCm APIs to provide a unified
/// interface for GPU operations.
pub trait GpuRuntime: Send + Sync {
    /// Get the runtime name.
    fn name(&self) -> &'static str;

    /// Initialize the runtime.
    fn init(&self) -> GpuResult<()>;

    /// Enumerate available devices.
    fn enumerate_devices(&self) -> GpuResult<Vec<DeviceInfo>>;

    /// Set the current device.
    fn set_device(&self, device: DeviceId) -> GpuResult<()>;

    /// Get the current device.
    fn get_device(&self) -> GpuResult<DeviceId>;

    /// Allocate device memory.
    fn malloc(&self, size: usize) -> GpuResult<DevicePtr>;

    /// Free device memory.
    fn free(&self, ptr: DevicePtr) -> GpuResult<()>;

    /// Set device memory to a value.
    fn memset(&self, ptr: DevicePtr, value: u8, size: usize) -> GpuResult<()>;

    /// Copy memory from host to device.
    fn memcpy_host_to_device(&self, dst: DevicePtr, src: *const u8, size: usize) -> GpuResult<()>;

    /// Copy memory from device to host.
    fn memcpy_device_to_host(&self, dst: *mut u8, src: DevicePtr, size: usize) -> GpuResult<()>;

    /// Copy memory between device buffers.
    fn memcpy_device_to_device(&self, dst: DevicePtr, src: DevicePtr, size: usize)
        -> GpuResult<()>;

    /// Create a stream.
    fn create_stream(&self) -> GpuResult<u64>;

    /// Destroy a stream.
    fn destroy_stream(&self, stream: u64) -> GpuResult<()>;

    /// Synchronize a stream.
    fn synchronize_stream(&self, stream: u64) -> GpuResult<()>;

    /// Synchronize the device.
    fn device_synchronize(&self) -> GpuResult<()>;

    /// Load a module from code.
    fn load_module(&self, code: &[u8]) -> GpuResult<u64>;

    /// Unload a module.
    fn unload_module(&self, module: u64) -> GpuResult<()>;

    /// Get a function from a module.
    fn get_function(&self, module: u64, name: &str) -> GpuResult<u64>;

    /// Launch a kernel.
    fn launch_kernel(
        &self,
        function: u64,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem: usize,
        stream: u64,
        args: &[u64],
    ) -> GpuResult<()>;

    /// Get free and total memory.
    fn memory_info(&self) -> GpuResult<(usize, usize)>;
}

/// Get the appropriate runtime for a device.
pub fn runtime_for_device(device: &DeviceInfo) -> Option<Box<dyn GpuRuntime>> {
    match device.kind {
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => Some(Box::new(cuda::CudaRuntime::new())),

        #[cfg(feature = "rocm")]
        DeviceKind::Rocm => Some(Box::new(rocm::RocmRuntime::new())),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_selection() {
        let mock = DeviceInfo::mock();

        // Mock device doesn't have a real runtime
        let runtime = runtime_for_device(&mock);

        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        assert!(runtime.is_none());
    }
}
