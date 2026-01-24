//! ARM NEON SIMD support
//!
//! This module provides SIMD vector types using ARM NEON intrinsics.
//! NEON is the SIMD extension for ARM processors, available on:
//! - Apple Silicon (M1/M2/M3)
//! - ARM Cortex-A series (A53, A57, A72, A73, A76, etc.)
//! - Raspberry Pi 3/4/5
//! - Most modern ARM-based devices
//!
//! # Vector Types
//!
//! NEON provides 128-bit SIMD registers (Q registers) that can hold:
//! - 4 x f32 (single precision floats)
//! - 2 x f64 (double precision floats)
//! - 4 x i32 (32-bit integers)
//! - 2 x i64 (64-bit integers)
//! - 8 x i16 (16-bit integers)
//! - 16 x i8 (8-bit integers)
//!
//! # Platform Support
//!
//! This module is only compiled on aarch64 (64-bit ARM) platforms.
//! It uses the `std::arch::aarch64` intrinsics.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================
// Vec4F32 - 4 x f32 (128-bit, NEON)
// ============================================================

/// 4 x f32 SIMD vector using ARM NEON.
///
/// This type provides 128-bit SIMD operations for single-precision
/// floating point values on ARM processors.
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::simd_neon::NeonVec4F32;
///
/// let a = NeonVec4F32::splat(2.0);
/// let b = NeonVec4F32::splat(3.0);
/// let c = a.add(b);
/// assert_eq!(c.sum(), 20.0); // 4 * 5.0
/// ```
#[cfg(target_arch = "aarch64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct NeonVec4F32 {
    inner: float32x4_t,
}

#[cfg(target_arch = "aarch64")]
impl NeonVec4F32 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: f32) -> Self {
        unsafe {
            Self {
                inner: vdupq_n_f32(x),
            }
        }
    }

    /// Create a new vector from 4 values.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        unsafe {
            let arr = [x, y, z, w];
            Self {
                inner: vld1q_f32(arr.as_ptr()),
            }
        }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Load from a slice (must have at least 4 elements).
    #[inline]
    pub fn load(slice: &[f32]) -> Self {
        assert!(slice.len() >= 4);
        unsafe {
            Self {
                inner: vld1q_f32(slice.as_ptr()),
            }
        }
    }

    /// Load from an aligned pointer (16-byte aligned).
    #[inline]
    pub unsafe fn load_aligned(ptr: *const f32) -> Self {
        Self {
            inner: vld1q_f32(ptr),
        }
    }

    /// Store to a mutable slice.
    #[inline]
    pub fn store(&self, slice: &mut [f32]) {
        assert!(slice.len() >= 4);
        unsafe {
            vst1q_f32(slice.as_mut_ptr(), self.inner);
        }
    }

    /// Store to an aligned pointer (16-byte aligned).
    #[inline]
    pub unsafe fn store_aligned(&self, ptr: *mut f32) {
        vst1q_f32(ptr, self.inner);
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        assert!(idx < 4);
        let mut arr = [0.0f32; 4];
        self.store(&mut arr);
        arr[idx]
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vaddq_f32(self.inner, other.inner),
            }
        }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vsubq_f32(self.inner, other.inner),
            }
        }
    }

    /// Element-wise multiplication.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vmulq_f32(self.inner, other.inner),
            }
        }
    }

    /// Element-wise division.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vdivq_f32(self.inner, other.inner),
            }
        }
    }

    /// Fused multiply-add: a * b + c.
    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        unsafe {
            Self {
                inner: vfmaq_f32(c.inner, self.inner, b.inner),
            }
        }
    }

    /// Element-wise negation.
    #[inline]
    pub fn neg(self) -> Self {
        unsafe {
            Self {
                inner: vnegq_f32(self.inner),
            }
        }
    }

    /// Element-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        unsafe {
            Self {
                inner: vabsq_f32(self.inner),
            }
        }
    }

    /// Element-wise square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        unsafe {
            Self {
                inner: vsqrtq_f32(self.inner),
            }
        }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vminq_f32(self.inner, other.inner),
            }
        }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vmaxq_f32(self.inner, other.inner),
            }
        }
    }

    /// Horizontal sum (add all elements).
    #[inline]
    pub fn sum(self) -> f32 {
        unsafe { vaddvq_f32(self.inner) }
    }

    /// Horizontal minimum.
    #[inline]
    pub fn hmin(self) -> f32 {
        unsafe { vminvq_f32(self.inner) }
    }

    /// Horizontal maximum.
    #[inline]
    pub fn hmax(self) -> f32 {
        unsafe { vmaxvq_f32(self.inner) }
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.mul(other).sum()
    }
}

#[cfg(target_arch = "aarch64")]
impl Default for NeonVec4F32 {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(target_arch = "aarch64")]
impl std::fmt::Debug for NeonVec4F32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0.0f32; 4];
        self.store(&mut arr);
        write!(f, "NeonVec4F32({:?})", arr)
    }
}

// ============================================================
// Vec2F64 - 2 x f64 (128-bit, NEON)
// ============================================================

/// 2 x f64 SIMD vector using ARM NEON.
///
/// This type provides 128-bit SIMD operations for double-precision
/// floating point values on ARM processors.
#[cfg(target_arch = "aarch64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct NeonVec2F64 {
    inner: float64x2_t,
}

#[cfg(target_arch = "aarch64")]
impl NeonVec2F64 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: f64) -> Self {
        unsafe {
            Self {
                inner: vdupq_n_f64(x),
            }
        }
    }

    /// Create a new vector from 2 values.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        unsafe {
            let arr = [x, y];
            Self {
                inner: vld1q_f64(arr.as_ptr()),
            }
        }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Load from a slice (must have at least 2 elements).
    #[inline]
    pub fn load(slice: &[f64]) -> Self {
        assert!(slice.len() >= 2);
        unsafe {
            Self {
                inner: vld1q_f64(slice.as_ptr()),
            }
        }
    }

    /// Store to a mutable slice.
    #[inline]
    pub fn store(&self, slice: &mut [f64]) {
        assert!(slice.len() >= 2);
        unsafe {
            vst1q_f64(slice.as_mut_ptr(), self.inner);
        }
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> f64 {
        assert!(idx < 2);
        let mut arr = [0.0f64; 2];
        self.store(&mut arr);
        arr[idx]
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vaddq_f64(self.inner, other.inner),
            }
        }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vsubq_f64(self.inner, other.inner),
            }
        }
    }

    /// Element-wise multiplication.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vmulq_f64(self.inner, other.inner),
            }
        }
    }

    /// Element-wise division.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vdivq_f64(self.inner, other.inner),
            }
        }
    }

    /// Fused multiply-add: a * b + c.
    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        unsafe {
            Self {
                inner: vfmaq_f64(c.inner, self.inner, b.inner),
            }
        }
    }

    /// Element-wise negation.
    #[inline]
    pub fn neg(self) -> Self {
        unsafe {
            Self {
                inner: vnegq_f64(self.inner),
            }
        }
    }

    /// Element-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        unsafe {
            Self {
                inner: vabsq_f64(self.inner),
            }
        }
    }

    /// Element-wise square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        unsafe {
            Self {
                inner: vsqrtq_f64(self.inner),
            }
        }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vminq_f64(self.inner, other.inner),
            }
        }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vmaxq_f64(self.inner, other.inner),
            }
        }
    }

    /// Horizontal sum (add all elements).
    #[inline]
    pub fn sum(self) -> f64 {
        unsafe { vaddvq_f64(self.inner) }
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.mul(other).sum()
    }
}

#[cfg(target_arch = "aarch64")]
impl Default for NeonVec2F64 {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(target_arch = "aarch64")]
impl std::fmt::Debug for NeonVec2F64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0.0f64; 2];
        self.store(&mut arr);
        write!(f, "NeonVec2F64({:?})", arr)
    }
}

// ============================================================
// Vec4I32 - 4 x i32 (128-bit, NEON)
// ============================================================

/// 4 x i32 SIMD vector using ARM NEON.
#[cfg(target_arch = "aarch64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct NeonVec4I32 {
    inner: int32x4_t,
}

#[cfg(target_arch = "aarch64")]
impl NeonVec4I32 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: i32) -> Self {
        unsafe {
            Self {
                inner: vdupq_n_s32(x),
            }
        }
    }

    /// Create a new vector from 4 values.
    #[inline]
    pub fn new(x: i32, y: i32, z: i32, w: i32) -> Self {
        unsafe {
            let arr = [x, y, z, w];
            Self {
                inner: vld1q_s32(arr.as_ptr()),
            }
        }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0)
    }

    /// Load from a slice.
    #[inline]
    pub fn load(slice: &[i32]) -> Self {
        assert!(slice.len() >= 4);
        unsafe {
            Self {
                inner: vld1q_s32(slice.as_ptr()),
            }
        }
    }

    /// Store to a mutable slice.
    #[inline]
    pub fn store(&self, slice: &mut [i32]) {
        assert!(slice.len() >= 4);
        unsafe {
            vst1q_s32(slice.as_mut_ptr(), self.inner);
        }
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> i32 {
        assert!(idx < 4);
        let mut arr = [0i32; 4];
        self.store(&mut arr);
        arr[idx]
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vaddq_s32(self.inner, other.inner),
            }
        }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vsubq_s32(self.inner, other.inner),
            }
        }
    }

    /// Element-wise multiplication.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vmulq_s32(self.inner, other.inner),
            }
        }
    }

    /// Element-wise negation.
    #[inline]
    pub fn neg(self) -> Self {
        unsafe {
            Self {
                inner: vnegq_s32(self.inner),
            }
        }
    }

    /// Element-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        unsafe {
            Self {
                inner: vabsq_s32(self.inner),
            }
        }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vminq_s32(self.inner, other.inner),
            }
        }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        unsafe {
            Self {
                inner: vmaxq_s32(self.inner, other.inner),
            }
        }
    }

    /// Horizontal sum (add all elements).
    #[inline]
    pub fn sum(self) -> i32 {
        unsafe { vaddvq_s32(self.inner) }
    }

    /// Horizontal minimum.
    #[inline]
    pub fn hmin(self) -> i32 {
        unsafe { vminvq_s32(self.inner) }
    }

    /// Horizontal maximum.
    #[inline]
    pub fn hmax(self) -> i32 {
        unsafe { vmaxvq_s32(self.inner) }
    }
}

#[cfg(target_arch = "aarch64")]
impl Default for NeonVec4I32 {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(target_arch = "aarch64")]
impl std::fmt::Debug for NeonVec4I32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0i32; 4];
        self.store(&mut arr);
        write!(f, "NeonVec4I32({:?})", arr)
    }
}

// ============================================================
// Runtime Detection
// ============================================================

/// Check if NEON is available on the current platform.
///
/// On aarch64, NEON is always available as part of the base ISA.
/// This function exists for API consistency with x86 SIMD detection.
#[inline]
pub fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is mandatory on aarch64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Check if SVE (Scalable Vector Extension) is available.
///
/// SVE provides variable-width SIMD (128-2048 bits) and is available on:
/// - ARM Neoverse N1, V1, V2
/// - Fujitsu A64FX (used in Fugaku supercomputer)
///
/// Note: SVE support is not yet implemented in this module.
#[inline]
pub fn is_sve_available() -> bool {
    // TODO: Implement SVE detection when we add SVE support
    false
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn test_vec4f32_splat() {
        let v = NeonVec4F32::splat(2.0);
        for i in 0..4 {
            assert_eq!(v.get(i), 2.0);
        }
    }

    #[test]
    fn test_vec4f32_add() {
        let a = NeonVec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = NeonVec4F32::new(5.0, 6.0, 7.0, 8.0);
        let c = a.add(b);
        assert_eq!(c.get(0), 6.0);
        assert_eq!(c.get(1), 8.0);
        assert_eq!(c.get(2), 10.0);
        assert_eq!(c.get(3), 12.0);
    }

    #[test]
    fn test_vec4f32_mul() {
        let a = NeonVec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = NeonVec4F32::new(2.0, 2.0, 2.0, 2.0);
        let c = a.mul(b);
        assert_eq!(c.get(0), 2.0);
        assert_eq!(c.get(1), 4.0);
        assert_eq!(c.get(2), 6.0);
        assert_eq!(c.get(3), 8.0);
    }

    #[test]
    fn test_vec4f32_fma() {
        let a = NeonVec4F32::splat(2.0);
        let b = NeonVec4F32::splat(3.0);
        let c = NeonVec4F32::splat(4.0);
        let result = a.fma(b, c);
        for i in 0..4 {
            assert_eq!(result.get(i), 10.0); // 2*3+4 = 10
        }
    }

    #[test]
    fn test_vec4f32_sum() {
        let v = NeonVec4F32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn test_vec4f32_dot() {
        let a = NeonVec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = NeonVec4F32::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a.dot(b), 10.0);
    }

    #[test]
    fn test_vec2f64_splat() {
        let v = NeonVec2F64::splat(3.14);
        assert_eq!(v.get(0), 3.14);
        assert_eq!(v.get(1), 3.14);
    }

    #[test]
    fn test_vec2f64_add() {
        let a = NeonVec2F64::new(1.0, 2.0);
        let b = NeonVec2F64::new(3.0, 4.0);
        let c = a.add(b);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(1), 6.0);
    }

    #[test]
    fn test_vec2f64_fma() {
        let a = NeonVec2F64::splat(2.0);
        let b = NeonVec2F64::splat(3.0);
        let c = NeonVec2F64::splat(4.0);
        let result = a.fma(b, c);
        assert_eq!(result.get(0), 10.0);
        assert_eq!(result.get(1), 10.0);
    }

    #[test]
    fn test_vec4i32_add() {
        let a = NeonVec4I32::new(1, 2, 3, 4);
        let b = NeonVec4I32::new(10, 20, 30, 40);
        let c = a.add(b);
        assert_eq!(c.get(0), 11);
        assert_eq!(c.get(1), 22);
        assert_eq!(c.get(2), 33);
        assert_eq!(c.get(3), 44);
    }

    #[test]
    fn test_vec4i32_sum() {
        let v = NeonVec4I32::new(1, 2, 3, 4);
        assert_eq!(v.sum(), 10);
    }

    #[test]
    fn test_is_neon_available() {
        assert!(is_neon_available());
    }
}
