//! BHC Numeric Library - Rust support
//!
//! High-performance numeric primitives for BHC.
//!
//! # Architecture
//!
//! Haskell Tensor/Vector/Matrix types are **defined in Haskell** (see
//! `hs/BHC/Numeric/*.hs`). This Rust crate provides the low-level
//! primitives that BHC-compiled code calls via FFI.
//!
//! # What belongs here
//!
//! - SIMD vector types (Vec4F32, Vec8F32, etc.) with platform intrinsics
//! - BLAS provider abstraction (OpenBLAS, MKL, Accelerate)
//! - Low-level memory layout for tensor backing stores
//! - FFI-exportable numeric primitives (dot, saxpy, gemm, etc.)
//!
//! # What does NOT belong here
//!
//! - High-level Tensor/Vector/Matrix API (that's Haskell)
//! - Fusion logic (that's BHC compiler)
//! - Type class instances (that's Haskell)
//!
//! # FFI Exports
//!
//! This crate exports C-ABI functions for BHC to call:
//! - `bhc_simd_dot_f32`, `bhc_simd_dot_f64` - Dot product
//! - `bhc_simd_sum_f32`, `bhc_simd_sum_f64` - Sum reduction
//! - `bhc_simd_saxpy` - SAXPY operation
//!
//! # Features
//!
//! - **SIMD**: SSE, AVX, AVX2, FMA intrinsics
//! - **BLAS**: Optional BLAS backend integration
//! - **Portable**: Scalar fallbacks for all operations

#![warn(missing_docs)]
#![allow(unsafe_code)] // SIMD requires unsafe

pub mod blas;
pub mod matrix;
pub mod simd;
pub mod tensor;
pub mod vector;
