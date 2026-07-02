//! Apple Metal compute runtime (macOS only).
//!
//! Unlike the CUDA and ROCm runtimes — which target hardware BHC's CI (and most
//! developer machines) does not have, and are therefore only mock-validated —
//! this runtime executes on the Apple GPU present on any Apple-silicon Mac. It
//! is the first GPU backend that can be *run and checked*, not just emitted.
//!
//! # Approach
//!
//! Metal Shading Language (MSL) is compiled at runtime from a source string via
//! `MTLDevice.newLibraryWithSource:` — the Metal framework's built-in shader
//! compiler, which does not require the offline Metal Toolchain component. This
//! mirrors the CUDA runtime's dynamic loading: no ahead-of-time toolchain step.
//!
//! On Apple silicon the CPU and GPU share memory, so buffers created with
//! `StorageModeShared` need no host↔device copies — the kernel reads and writes
//! host-visible memory directly.
//!
//! The MSL executed here is produced by [`crate::codegen::metal`]; this module
//! is the execution half that was previously missing (`runtime/` had only
//! `cuda.rs` and `rocm.rs`).

use metal::{
    CompileOptions, ComputePipelineState, Device, Function, Library, MTLResourceOptions, MTLSize,
};

/// A Metal compute runtime bound to the system-default GPU.
///
/// Holds the `MTLDevice` and a command queue. Construct with [`MetalRuntime::new`],
/// which returns `None` when no Metal device is available.
pub struct MetalRuntime {
    device: Device,
    queue: metal::CommandQueue,
}

impl MetalRuntime {
    /// Create a runtime on the system-default Metal device.
    ///
    /// Returns `None` if the machine has no Metal-capable GPU.
    #[must_use]
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        Some(Self { device, queue })
    }

    /// The GPU's product name (e.g. `"Apple M4"`).
    #[must_use]
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Compile `msl` and build a compute pipeline for the kernel named `entry`.
    ///
    /// # Errors
    ///
    /// Returns the compiler/linker diagnostic string if the MSL fails to compile
    /// or the entry point is missing.
    pub fn build_pipeline(&self, msl: &str, entry: &str) -> Result<ComputePipelineState, String> {
        let library: Library = self
            .device
            .new_library_with_source(msl, &CompileOptions::new())?;
        let function: Function = library.get_function(entry, None)?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
    }

    /// Compile `msl`, run the kernel `entry` over `out_len` threads, and return
    /// the output buffer as `f32`s.
    ///
    /// The `f32` input slices are bound to buffers `0..inputs.len()`; a freshly
    /// allocated output buffer of `out_len` elements is bound to the next index.
    /// This matches the buffer layout emitted by [`crate::codegen::metal`]
    /// (`device const float* inN [[buffer(N)]]`, output last) for element-wise
    /// and `zipWith`-style kernels indexed by `thread_position_in_grid`.
    ///
    /// All buffers use shared (unified) storage, so there are no explicit copies.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic string on MSL compilation failure.
    pub fn run_elementwise_f32(
        &self,
        msl: &str,
        entry: &str,
        inputs: &[&[f32]],
        out_len: usize,
    ) -> Result<Vec<f32>, String> {
        let pipeline = self.build_pipeline(msl, entry)?;
        let shared = MTLResourceOptions::StorageModeShared;
        let elem = std::mem::size_of::<f32>() as u64;

        let in_bufs: Vec<metal::Buffer> = inputs
            .iter()
            .map(|data| {
                self.device.new_buffer_with_data(
                    data.as_ptr().cast::<std::ffi::c_void>(),
                    elem * data.len() as u64,
                    shared,
                )
            })
            .collect();
        let out_buf = self.device.new_buffer(elem * out_len as u64, shared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        for (i, b) in in_bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_buffer(in_bufs.len() as u64, Some(&out_buf), 0);

        // Apple silicon supports non-uniform threadgroups, so dispatch the exact
        // thread count and let the driver size the final threadgroup.
        let tg = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        enc.dispatch_threads(MTLSize::new(out_len as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents().cast::<f32>();
        // SAFETY: `out_buf` holds `out_len` contiguous f32s in shared memory and
        // the command buffer has completed, so the writes are visible.
        let out = unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec();
        Ok(out)
    }
}

/// Enumerate the system-default Apple GPU as a [`DeviceInfo`].
///
/// Returns a single-element vector describing the Metal device
/// (`DeviceKind::Metal`, real product name, unified memory), or an empty vector
/// if no Metal device is available. This is what lets [`crate::available_devices`]
/// surface a *real* GPU on macOS instead of falling back to the mock device — so
/// the `DeviceKind::Metal` branch of the code generator is reached with actual
/// hardware.
///
/// Non-Metal-specific limits reuse [`DeviceInfo::mock`]'s conservative defaults;
/// querying exact device limits is a follow-up.
#[must_use]
pub fn enumerate_devices() -> Vec<crate::device::DeviceInfo> {
    let Some(rt) = MetalRuntime::new() else {
        return Vec::new();
    };
    let mut info = crate::device::DeviceInfo::mock();
    info.kind = crate::device::DeviceKind::Metal;
    info.name = rt.device_name();
    info.warp_size = 32; // Apple GPU SIMD-group width
    info.unified_memory = true; // Apple silicon shares CPU/GPU memory
    info.pci_bus_id = None; // integrated GPU
    info.compute_capability = (3, 0); // Metal 3
    vec![info]
}

#[cfg(test)]
mod tests {
    use super::*;

    const VADD: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* out     [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}";

    const SCALE: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void scale2(device const float* in0 [[buffer(0)]],
                   device float* out0       [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    out0[i] = in0[i] * 2.0f;
}";

    /// End-to-end: compile MSL at runtime and execute `zipWith (+)` on the GPU.
    #[test]
    fn vadd_runs_on_gpu() {
        let Some(rt) = MetalRuntime::new() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let n = 1024;
        let a: Vec<f32> = vec![2.0; n];
        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let out = rt
            .run_elementwise_f32(VADD, "vadd", &[&a, &b], n)
            .expect("kernel run");
        assert_eq!(out.len(), n);
        for i in 0..n {
            assert_eq!(out[i], a[i] + b[i], "mismatch at {i}");
        }
    }

    /// End-to-end: a single-input `map (*2)` element-wise kernel.
    #[test]
    fn scale_map_runs_on_gpu() {
        let Some(rt) = MetalRuntime::new() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let out = rt
            .run_elementwise_f32(SCALE, "scale2", &[&input], input.len())
            .expect("kernel run");
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, input[i] * 2.0, "mismatch at {i}");
        }
    }

    /// Bridge: the existing `codegen::metal` MSL generator produces a kernel that
    /// compiles and runs correctly on the GPU. This validates the code generator
    /// against real hardware — previously it was only checked in mock mode.
    #[test]
    fn codegen_metal_kernel_runs_on_gpu() {
        let Some(rt) = MetalRuntime::new() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        use crate::codegen::metal as mslgen;
        use crate::codegen::KernelParams;
        use crate::device::{DeviceInfo, DeviceKind};
        use bhc_tensor_ir::DType;

        let mut device = DeviceInfo::mock();
        device.kind = DeviceKind::Metal;
        let params = KernelParams {
            name: "square_k".to_string(),
            inputs: vec![],
            outputs: vec![],
            shared_memory: 0,
            block_size: 256,
        };
        let header = mslgen::generate_module_header("square_k", &device);
        let kernel = mslgen::generate_elementwise_kernel(&params, "val * val", DType::Float32)
            .expect("codegen");
        let msl = format!("{header}{kernel}");

        let input: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let out = rt
            .run_elementwise_f32(&msl, "square_k", &[&input], input.len())
            .expect("codegen'd kernel run");
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, input[i] * input[i], "mismatch at {i}");
        }
    }

    /// The Apple GPU is enumerated as a real `Metal` device (not the mock), and
    /// `available_devices()` surfaces it.
    #[test]
    fn enumerates_real_metal_device() {
        use crate::device::DeviceKind;
        let devices = enumerate_devices();
        if devices.is_empty() {
            eprintln!("no Metal device; skipping");
            return;
        }
        assert_eq!(devices[0].kind, DeviceKind::Metal);
        assert!(!devices[0].name.is_empty());
        assert!(devices[0].unified_memory);
        // The top-level enumeration must include a non-mock device now.
        let all = crate::available_devices();
        assert!(all.iter().any(|d| d.kind == DeviceKind::Metal));
    }

    /// A malformed MSL source must surface the compiler diagnostic, not panic.
    #[test]
    fn bad_msl_reports_error() {
        let Some(rt) = MetalRuntime::new() else {
            return;
        };
        let err = rt
            .run_elementwise_f32("this is not valid MSL", "nope", &[], 1)
            .unwrap_err();
        assert!(!err.is_empty());
    }
}
