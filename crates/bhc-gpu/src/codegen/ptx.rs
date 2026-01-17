//! PTX code generation for NVIDIA GPUs.
//!
//! This module generates PTX (Parallel Thread Execution) assembly from
//! Tensor IR kernels. PTX is NVIDIA's virtual instruction set that gets
//! JIT-compiled by the CUDA driver to target-specific machine code.
//!
//! # PTX Overview
//!
//! PTX is a pseudo-assembly language with:
//! - Virtual registers (unlimited, driver allocates physical)
//! - Typed operations (f32.add, s32.mul, etc.)
//! - Special registers (%tid, %ntid, %ctaid, etc.)
//! - Memory spaces (global, shared, local, const)
//!
//! # Example PTX
//!
//! ```ptx
//! .version 7.0
//! .target sm_80
//! .address_size 64
//!
//! .visible .entry kernel_add(
//!     .param .u64 a,
//!     .param .u64 b,
//!     .param .u64 c,
//!     .param .u64 n
//! ) {
//!     .reg .u32 %tid, %idx;
//!     .reg .u64 %addr_a, %addr_b, %addr_c;
//!     .reg .f32 %val_a, %val_b, %result;
//!
//!     mov.u32 %tid, %tid.x;
//!     // ... compute and store
//!     ret;
//! }
//! ```

use super::{dtype_to_gpu_type, KernelParams};
use crate::device::DeviceInfo;
use crate::kernel::CompiledModule;
use crate::GpuResult;
use bhc_tensor_ir::{BinaryOp, DType, Kernel, KernelBody, TensorOp, UnaryOp};
use std::fmt::Write;

/// PTX version to target.
const PTX_VERSION: &str = "7.0";

/// Generate PTX module header.
pub fn generate_module_header(name: &str, device: &DeviceInfo) -> String {
    let arch = device.arch_name();
    format!(
        ".version {}\n\
         .target {}\n\
         .address_size 64\n\
         \n\
         // Module: {}\n\
         // Device: {}\n\n",
        PTX_VERSION, arch, name, device.name
    )
}

/// Compile a Tensor IR kernel to PTX.
pub fn compile_kernel(kernel: &Kernel, device: &DeviceInfo) -> GpuResult<CompiledModule> {
    let params = KernelParams::from_kernel(kernel);
    let mut code = generate_module_header(&params.name, device);

    // Generate kernel entry point
    generate_kernel_entry(&mut code, &params, kernel)?;

    let mut module = CompiledModule::from_text(
        params.name.clone(),
        code,
        device.arch_name(),
    );
    module.add_entry_point(params.name);

    Ok(module)
}

/// Generate a kernel entry point.
fn generate_kernel_entry(
    code: &mut String,
    params: &KernelParams,
    kernel: &Kernel,
) -> GpuResult<()> {
    // Entry point signature
    writeln!(code, ".visible .entry {}(", params.name).unwrap();

    // Parameters
    let all_params: Vec<_> = params
        .inputs
        .iter()
        .chain(params.outputs.iter())
        .collect();

    for (i, param) in all_params.iter().enumerate() {
        let sep = if i < all_params.len() - 1 { "," } else { "" };
        writeln!(code, "    .param .u64 ptr_{}{}  // {}", param.name, sep, param.name).unwrap();
    }

    // Add size parameter
    writeln!(code, ") {{").unwrap();

    // Register declarations
    writeln!(code, "    // Register declarations").unwrap();
    writeln!(code, "    .reg .u32 %tid, %ntid, %ctaid;").unwrap();
    writeln!(code, "    .reg .u64 %idx, %n;").unwrap();
    writeln!(code, "    .reg .pred %p;").unwrap();
    writeln!(code).unwrap();

    // Thread index calculation
    writeln!(code, "    // Calculate global thread index").unwrap();
    writeln!(code, "    mov.u32 %tid, %tid.x;").unwrap();
    writeln!(code, "    mov.u32 %ntid, %ntid.x;").unwrap();
    writeln!(code, "    mov.u32 %ctaid, %ctaid.x;").unwrap();
    writeln!(code, "    mad.wide.u32 %idx, %ctaid, %ntid, %tid;").unwrap();
    writeln!(code).unwrap();

    // Generate kernel body based on operation
    match &kernel.body {
        KernelBody::Fused(ops) => {
            generate_fused_ops(code, ops, params)?;
        }
        KernelBody::LoopNest(nest) => {
            generate_loop_nest(code, nest, params)?;
        }
    }

    // Return
    writeln!(code, "    ret;").unwrap();
    writeln!(code, "}}").unwrap();

    Ok(())
}

/// Generate code for fused operations.
fn generate_fused_ops(
    code: &mut String,
    ops: &[TensorOp],
    _params: &KernelParams,
) -> GpuResult<()> {
    writeln!(code, "    // Fused operations").unwrap();

    for (i, op) in ops.iter().enumerate() {
        match op {
            TensorOp::Unary(unary_op, input) => {
                let dtype = input.meta.dtype;
                let _ty = dtype_to_gpu_type(dtype);
                writeln!(code, "    // Unary: {:?}", unary_op).unwrap();
                generate_unary_op(code, *unary_op, dtype, i)?;
            }
            TensorOp::Binary(binary_op, left, _right) => {
                let dtype = left.meta.dtype;
                writeln!(code, "    // Binary: {:?}", binary_op).unwrap();
                generate_binary_op(code, *binary_op, dtype, i)?;
            }
            TensorOp::Map(map_fn, _input) => {
                writeln!(code, "    // Map: {}", map_fn.name.as_str()).unwrap();
            }
            TensorOp::ZipWith(zip_fn, _left, _right) => {
                writeln!(code, "    // ZipWith: {}", zip_fn.name.as_str()).unwrap();
            }
            TensorOp::Reduce(reduce_op, axis, _input) => {
                writeln!(code, "    // Reduce: {:?} axis={}", reduce_op, axis.0).unwrap();
            }
            _ => {
                writeln!(code, "    // Unsupported op").unwrap();
            }
        }
    }

    Ok(())
}

/// Generate code for a loop nest.
fn generate_loop_nest(
    code: &mut String,
    _nest: &bhc_tensor_ir::LoopNest,
    _params: &KernelParams,
) -> GpuResult<()> {
    writeln!(code, "    // Loop nest").unwrap();
    // TODO: Implement proper loop nest code generation
    Ok(())
}

/// Generate a unary operation.
fn generate_unary_op(
    code: &mut String,
    op: UnaryOp,
    dtype: DType,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);
    let reg_in = format!("%in{}", idx);
    let reg_out = format!("%out{}", idx);

    match op {
        UnaryOp::Neg => {
            writeln!(code, "    neg.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Abs => {
            writeln!(code, "    abs.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Sqrt => {
            writeln!(code, "    sqrt.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Rsqrt => {
            writeln!(code, "    rsqrt.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Exp => {
            writeln!(code, "    ex2.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Log => {
            writeln!(code, "    lg2.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Sin => {
            writeln!(code, "    sin.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Cos => {
            writeln!(code, "    cos.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        _ => {
            writeln!(code, "    // Unsupported unary: {:?}", op).unwrap();
        }
    }

    Ok(())
}

/// Generate a binary operation.
fn generate_binary_op(
    code: &mut String,
    op: BinaryOp,
    dtype: DType,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);
    let reg_a = format!("%a{}", idx);
    let reg_b = format!("%b{}", idx);
    let reg_out = format!("%out{}", idx);

    match op {
        BinaryOp::Add => {
            writeln!(code, "    add.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Sub => {
            writeln!(code, "    sub.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Mul => {
            writeln!(code, "    mul.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Div => {
            writeln!(code, "    div.approx.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Max => {
            writeln!(code, "    max.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Min => {
            writeln!(code, "    min.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        _ => {
            writeln!(code, "    // Unsupported binary: {:?}", op).unwrap();
        }
    }

    Ok(())
}

/// Generate a simple elementwise kernel.
pub fn generate_elementwise_kernel(
    name: &str,
    op: BinaryOp,
    dtype: DType,
    device: &DeviceInfo,
) -> String {
    let ty = dtype_to_gpu_type(dtype);
    let arch = device.arch_name();

    format!(
        ".version {PTX_VERSION}\n\
         .target {arch}\n\
         .address_size 64\n\
         \n\
         .visible .entry {name}(\n\
             .param .u64 a,\n\
             .param .u64 b,\n\
             .param .u64 c,\n\
             .param .u64 n\n\
         ) {{\n\
             .reg .u32 %tid, %ntid, %ctaid;\n\
             .reg .u64 %idx, %len, %off;\n\
             .reg .{ty} %va, %vb, %vc;\n\
             .reg .pred %p;\n\
         \n\
             mov.u32 %tid, %tid.x;\n\
             mov.u32 %ntid, %ntid.x;\n\
             mov.u32 %ctaid, %ctaid.x;\n\
             mad.wide.u32 %idx, %ctaid, %ntid, %tid;\n\
         \n\
             ld.param.u64 %len, [n];\n\
             setp.ge.u64 %p, %idx, %len;\n\
             @%p bra done;\n\
         \n\
             shl.b64 %off, %idx, 2;\n\
         \n\
             ld.param.u64 %va, [a];\n\
             add.u64 %va, %va, %off;\n\
             ld.global.{ty} %va, [%va];\n\
         \n\
             ld.param.u64 %vb, [b];\n\
             add.u64 %vb, %vb, %off;\n\
             ld.global.{ty} %vb, [%vb];\n\
         \n\
             {op_code}\n\
         \n\
             ld.param.u64 %vc, [c];\n\
             add.u64 %vc, %vc, %off;\n\
             st.global.{ty} [%vc], %vc;\n\
         \n\
         done:\n\
             ret;\n\
         }}\n",
        op_code = match op {
            BinaryOp::Add => format!("add.{ty} %vc, %va, %vb;"),
            BinaryOp::Sub => format!("sub.{ty} %vc, %va, %vb;"),
            BinaryOp::Mul => format!("mul.{ty} %vc, %va, %vb;"),
            BinaryOp::Div => format!("div.approx.{ty} %vc, %va, %vb;"),
            _ => "// unsupported op".to_string(),
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_module_header() {
        let device = DeviceInfo::mock();
        let header = generate_module_header("test_kernel", &device);

        assert!(header.contains(".version 7.0"));
        assert!(header.contains(".target"));
        assert!(header.contains("test_kernel"));
    }

    #[test]
    fn test_generate_elementwise_kernel() {
        let device = DeviceInfo::mock();
        let ptx = generate_elementwise_kernel("add_kernel", BinaryOp::Add, DType::Float32, &device);

        assert!(ptx.contains(".visible .entry add_kernel"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("ret;"));
    }
}
