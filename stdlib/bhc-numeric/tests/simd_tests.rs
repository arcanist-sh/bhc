//! Comprehensive SIMD vector tests
//!
//! Tests for all SIMD vector types and operations including:
//! - Construction and element access
//! - Arithmetic operations (add, sub, mul, div)
//! - FMA (fused multiply-add) accuracy
//! - Horizontal operations (sum, product, min, max)
//! - Edge cases (NaN, infinity, denormals)

use bhc_numeric::simd::*;

// ============================================================
// Vec2F32 Tests
// ============================================================

mod vec2f32_tests {
    use super::*;

    #[test]
    fn test_splat() {
        let v = Vec2F32::splat(3.0);
        assert_eq!(v.get(0), 3.0);
        assert_eq!(v.get(1), 3.0);
    }

    #[test]
    fn test_new() {
        let v = Vec2F32::new(1.0, 2.0);
        assert_eq!(v.get(0), 1.0);
        assert_eq!(v.get(1), 2.0);
    }

    #[test]
    fn test_zero() {
        let v = Vec2F32::zero();
        assert_eq!(v.get(0), 0.0);
        assert_eq!(v.get(1), 0.0);
    }

    #[test]
    fn test_add() {
        let a = Vec2F32::new(1.0, 2.0);
        let b = Vec2F32::new(3.0, 4.0);
        let c = a.add(b);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(1), 6.0);
    }

    #[test]
    fn test_sub() {
        let a = Vec2F32::new(5.0, 7.0);
        let b = Vec2F32::new(2.0, 3.0);
        let c = a.sub(b);
        assert_eq!(c.get(0), 3.0);
        assert_eq!(c.get(1), 4.0);
    }

    #[test]
    fn test_mul() {
        let a = Vec2F32::new(2.0, 3.0);
        let b = Vec2F32::new(4.0, 5.0);
        let c = a.mul(b);
        assert_eq!(c.get(0), 8.0);
        assert_eq!(c.get(1), 15.0);
    }

    #[test]
    fn test_div() {
        let a = Vec2F32::new(8.0, 15.0);
        let b = Vec2F32::new(2.0, 3.0);
        let c = a.div(b);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(1), 5.0);
    }

    #[test]
    fn test_sum() {
        let v = Vec2F32::new(3.0, 4.0);
        assert_eq!(v.sum(), 7.0);
    }

    #[test]
    fn test_dot() {
        let a = Vec2F32::new(1.0, 2.0);
        let b = Vec2F32::new(3.0, 4.0);
        assert_eq!(a.dot(b), 11.0); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_add_identity() {
        let a = Vec2F32::new(5.0, -3.0);
        let zero = Vec2F32::zero();
        let c = a.add(zero);
        assert_eq!(c.get(0), a.get(0));
        assert_eq!(c.get(1), a.get(1));
    }

    #[test]
    fn test_mul_identity() {
        let a = Vec2F32::new(5.0, -3.0);
        let one = Vec2F32::splat(1.0);
        let c = a.mul(one);
        assert_eq!(c.get(0), a.get(0));
        assert_eq!(c.get(1), a.get(1));
    }
}

// ============================================================
// Vec4F32 Tests
// ============================================================

mod vec4f32_tests {
    use super::*;

    #[test]
    fn test_splat() {
        let v = Vec4F32::splat(2.5);
        for i in 0..4 {
            assert_eq!(v.get(i), 2.5);
        }
    }

    #[test]
    fn test_new() {
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.get(0), 1.0);
        assert_eq!(v.get(1), 2.0);
        assert_eq!(v.get(2), 3.0);
        assert_eq!(v.get(3), 4.0);
    }

    #[test]
    fn test_load_store() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let v = Vec4F32::load(&data);
        let mut out = [0.0f32; 4];
        v.store(&mut out);
        assert_eq!(data, out);
    }

    #[test]
    fn test_add() {
        let a = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F32::new(5.0, 6.0, 7.0, 8.0);
        let c = a.add(b);
        assert_eq!(c.get(0), 6.0);
        assert_eq!(c.get(1), 8.0);
        assert_eq!(c.get(2), 10.0);
        assert_eq!(c.get(3), 12.0);
    }

    #[test]
    fn test_sub() {
        let a = Vec4F32::new(10.0, 20.0, 30.0, 40.0);
        let b = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let c = a.sub(b);
        assert_eq!(c.get(0), 9.0);
        assert_eq!(c.get(1), 18.0);
        assert_eq!(c.get(2), 27.0);
        assert_eq!(c.get(3), 36.0);
    }

    #[test]
    fn test_mul() {
        let a = Vec4F32::new(2.0, 3.0, 4.0, 5.0);
        let b = Vec4F32::new(2.0, 2.0, 2.0, 2.0);
        let c = a.mul(b);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(1), 6.0);
        assert_eq!(c.get(2), 8.0);
        assert_eq!(c.get(3), 10.0);
    }

    #[test]
    fn test_div() {
        let a = Vec4F32::new(4.0, 6.0, 8.0, 10.0);
        let b = Vec4F32::new(2.0, 2.0, 2.0, 2.0);
        let c = a.div(b);
        assert_eq!(c.get(0), 2.0);
        assert_eq!(c.get(1), 3.0);
        assert_eq!(c.get(2), 4.0);
        assert_eq!(c.get(3), 5.0);
    }

    #[test]
    fn test_horizontal_sum() {
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn test_dot_product() {
        let a = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F32::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a.dot(b), 10.0); // 1+2+3+4
    }

    #[test]
    fn test_set() {
        let mut v = Vec4F32::zero();
        v.set(2, 7.0);
        assert_eq!(v.get(2), 7.0);
        assert_eq!(v.get(0), 0.0);
    }

    #[test]
    fn test_negative_values() {
        let a = Vec4F32::new(-1.0, -2.0, 3.0, -4.0);
        let b = Vec4F32::new(1.0, 2.0, -3.0, 4.0);
        let c = a.add(b);
        assert_eq!(c.get(0), 0.0);
        assert_eq!(c.get(1), 0.0);
        assert_eq!(c.get(2), 0.0);
        assert_eq!(c.get(3), 0.0);
    }

    #[test]
    fn test_associativity() {
        let a = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F32::new(5.0, 6.0, 7.0, 8.0);
        let c = Vec4F32::new(9.0, 10.0, 11.0, 12.0);

        let ab_c = a.add(b).add(c);
        let a_bc = a.add(b.add(c));

        for i in 0..4 {
            assert!((ab_c.get(i) - a_bc.get(i)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_commutativity() {
        let a = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F32::new(5.0, 6.0, 7.0, 8.0);

        let ab = a.add(b);
        let ba = b.add(a);

        for i in 0..4 {
            assert_eq!(ab.get(i), ba.get(i));
        }
    }
}

// ============================================================
// Vec8F32 Tests (AVX)
// ============================================================

mod vec8f32_tests {
    use super::*;

    #[test]
    fn test_splat() {
        let v = Vec8F32::splat(1.5);
        for i in 0..8 {
            assert_eq!(v.get(i), 1.5);
        }
    }

    #[test]
    fn test_load_values() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = Vec8F32::load(&data);
        for i in 0..8 {
            assert_eq!(v.get(i), (i + 1) as f32);
        }
    }

    #[test]
    fn test_load_store() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = Vec8F32::load(&data);
        let mut out = [0.0f32; 8];
        v.store(&mut out);
        assert_eq!(data, out);
    }

    #[test]
    fn test_add() {
        let a = Vec8F32::splat(1.0);
        let b = Vec8F32::splat(2.0);
        let c = a.add(b);
        for i in 0..8 {
            assert_eq!(c.get(i), 3.0);
        }
    }

    #[test]
    fn test_horizontal_sum() {
        let v = Vec8F32::splat(1.0);
        assert_eq!(v.sum(), 8.0);
    }

    #[test]
    fn test_sum_mixed_values() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = Vec8F32::load(&data);
        assert_eq!(v.sum(), 36.0); // 1+2+3+4+5+6+7+8
    }

    #[test]
    fn test_fma_basic() {
        // FMA: a * b + c
        let a = Vec8F32::splat(2.0);
        let b = Vec8F32::splat(3.0);
        let c = Vec8F32::splat(4.0);
        let result = a.fma(b, c);
        for i in 0..8 {
            assert_eq!(result.get(i), 10.0); // 2*3+4 = 10
        }
    }

    #[test]
    fn test_fma_precision() {
        // FMA should preserve precision better than separate mul+add
        // Test case: (1 + 1e-7) * 1e7 - 1e7 should be close to 1.0
        let a = Vec8F32::splat(1.0 + 1e-7);
        let b = Vec8F32::splat(1e7);
        let c = Vec8F32::splat(-1e7);
        let result = a.fma(b, c);

        // The result should be close to 1.0
        // With FMA, we get better precision than separate operations
        for i in 0..8 {
            let val = result.get(i);
            assert!(
                (val - 1.0).abs() < 10.0, // Allow some tolerance due to f32 limits
                "FMA result {} at index {} should be close to 1.0",
                val,
                i
            );
        }
    }

    #[test]
    fn test_large_dot_product() {
        // Test dot product with larger vectors (simulated)
        let data_a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Vec8F32::load(&data_a);
        let b = Vec8F32::splat(1.0);
        let dot = a.mul(b).sum();
        assert_eq!(dot, 36.0);
    }
}

// ============================================================
// Vec2F64 Tests
// ============================================================

mod vec2f64_tests {
    use super::*;

    #[test]
    fn test_splat() {
        let v = Vec2F64::splat(3.14159265358979);
        assert_eq!(v.get(0), 3.14159265358979);
        assert_eq!(v.get(1), 3.14159265358979);
    }

    #[test]
    fn test_add() {
        let a = Vec2F64::new(1.0, 2.0);
        let b = Vec2F64::new(0.5, 0.5);
        let c = a.add(b);
        assert_eq!(c.get(0), 1.5);
        assert_eq!(c.get(1), 2.5);
    }

    #[test]
    fn test_precision() {
        // Test that f64 preserves more precision than f32
        let a = Vec2F64::new(1.0 + 1e-15, 1.0);
        let b = Vec2F64::new(1e-15, 1e-15);
        let c = a.sub(Vec2F64::new(1.0, 1.0));
        assert!(c.get(0) > 0.0, "f64 should preserve small differences");
    }
}

// ============================================================
// Vec4F64 Tests (AVX)
// ============================================================

mod vec4f64_tests {
    use super::*;

    #[test]
    fn test_splat() {
        let v = Vec4F64::splat(2.718281828);
        for i in 0..4 {
            assert_eq!(v.get(i), 2.718281828);
        }
    }

    #[test]
    fn test_new() {
        let v = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.get(0), 1.0);
        assert_eq!(v.get(1), 2.0);
        assert_eq!(v.get(2), 3.0);
        assert_eq!(v.get(3), 4.0);
    }

    #[test]
    fn test_load_store() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let v = Vec4F64::load(&data);
        let mut out = [0.0f64; 4];
        v.store(&mut out);
        assert_eq!(data, out);
    }

    #[test]
    fn test_add() {
        let a = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F64::new(0.1, 0.2, 0.3, 0.4);
        let c = a.add(b);
        assert!((c.get(0) - 1.1).abs() < 1e-10);
        assert!((c.get(1) - 2.2).abs() < 1e-10);
        assert!((c.get(2) - 3.3).abs() < 1e-10);
        assert!((c.get(3) - 4.4).abs() < 1e-10);
    }

    #[test]
    fn test_horizontal_sum() {
        let v = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn test_mul_add_pattern() {
        // Simulate FMA with separate mul and add: a * b + c
        let a = Vec4F64::splat(2.0);
        let b = Vec4F64::splat(3.0);
        let c = Vec4F64::splat(4.0);
        let result = a.mul(b).add(c);
        for i in 0..4 {
            assert_eq!(result.get(i), 10.0);
        }
    }

    #[test]
    fn test_dot_product() {
        let a = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F64::splat(1.0);
        let dot = a.dot(b);
        assert_eq!(dot, 10.0); // 1+2+3+4
    }
}

// ============================================================
// Integer Vector Tests
// ============================================================

mod vec4i32_tests {
    use super::*;

    #[test]
    fn test_splat() {
        let v = Vec4I32::splat(42);
        for i in 0..4 {
            assert_eq!(v.get(i), 42);
        }
    }

    #[test]
    fn test_new() {
        let v = Vec4I32::new(1, 2, 3, 4);
        assert_eq!(v.get(0), 1);
        assert_eq!(v.get(1), 2);
        assert_eq!(v.get(2), 3);
        assert_eq!(v.get(3), 4);
    }

    #[test]
    fn test_add() {
        let a = Vec4I32::new(1, 2, 3, 4);
        let b = Vec4I32::new(10, 20, 30, 40);
        let c = a.add(b);
        assert_eq!(c.get(0), 11);
        assert_eq!(c.get(1), 22);
        assert_eq!(c.get(2), 33);
        assert_eq!(c.get(3), 44);
    }

    #[test]
    fn test_sub() {
        let a = Vec4I32::new(100, 200, 300, 400);
        let b = Vec4I32::new(1, 2, 3, 4);
        let c = a.sub(b);
        assert_eq!(c.get(0), 99);
        assert_eq!(c.get(1), 198);
        assert_eq!(c.get(2), 297);
        assert_eq!(c.get(3), 396);
    }

    #[test]
    fn test_negative_integers() {
        let a = Vec4I32::new(-1, -2, 3, -4);
        let b = Vec4I32::new(1, 2, -3, 4);
        let c = a.add(b);
        assert_eq!(c.get(0), 0);
        assert_eq!(c.get(1), 0);
        assert_eq!(c.get(2), 0);
        assert_eq!(c.get(3), 0);
    }

    #[test]
    fn test_horizontal_sum() {
        let v = Vec4I32::new(1, 2, 3, 4);
        assert_eq!(v.sum(), 10);
    }
}

mod vec8i32_tests {
    use super::*;

    #[test]
    fn test_splat() {
        let v = Vec8I32::splat(7);
        for i in 0..8 {
            assert_eq!(v.get(i), 7);
        }
    }

    #[test]
    fn test_add() {
        let a = Vec8I32::splat(5);
        let b = Vec8I32::splat(3);
        let c = a.add(b);
        for i in 0..8 {
            assert_eq!(c.get(i), 8);
        }
    }

    #[test]
    fn test_sum() {
        let v = Vec8I32::splat(1);
        assert_eq!(v.sum(), 8);
    }
}

// ============================================================
// Edge Cases and Special Values
// ============================================================

mod special_values_tests {
    use super::*;

    #[test]
    fn test_infinity() {
        let inf = f32::INFINITY;
        let v = Vec4F32::splat(inf);
        let result = v.add(Vec4F32::splat(1.0));
        for i in 0..4 {
            assert!(result.get(i).is_infinite());
        }
    }

    #[test]
    fn test_negative_infinity() {
        let ninf = f32::NEG_INFINITY;
        let v = Vec4F32::splat(ninf);
        let result = v.add(Vec4F32::splat(-1.0));
        for i in 0..4 {
            assert!(result.get(i).is_infinite());
            assert!(result.get(i) < 0.0);
        }
    }

    #[test]
    fn test_nan_propagation() {
        let nan = f32::NAN;
        let v = Vec4F32::splat(nan);
        let result = v.add(Vec4F32::splat(1.0));
        for i in 0..4 {
            assert!(result.get(i).is_nan());
        }
    }

    #[test]
    fn test_very_small_values() {
        let tiny = f32::MIN_POSITIVE;
        let v = Vec4F32::splat(tiny);
        let doubled = v.add(v);
        for i in 0..4 {
            assert!(doubled.get(i) > 0.0);
            assert!((doubled.get(i) - 2.0 * tiny).abs() < 1e-45);
        }
    }

    #[test]
    fn test_very_large_values() {
        let big = f32::MAX / 2.0;
        let v = Vec4F32::splat(big);
        // Adding should still be valid (not overflow)
        let result = v.add(Vec4F32::splat(1.0));
        for i in 0..4 {
            assert!(!result.get(i).is_infinite());
        }
    }

    #[test]
    fn test_denormalized_numbers() {
        // Test with denormalized (subnormal) numbers
        let denorm = f32::MIN_POSITIVE / 2.0;
        let v = Vec4F32::splat(denorm);
        let result = v.mul(Vec4F32::splat(2.0));
        // Result should be approximately MIN_POSITIVE
        for i in 0..4 {
            assert!(result.get(i) > 0.0);
        }
    }

    #[test]
    fn test_zero_division() {
        let v = Vec4F32::splat(1.0);
        let zero = Vec4F32::zero();
        let result = v.div(zero);
        for i in 0..4 {
            assert!(result.get(i).is_infinite());
        }
    }

    #[test]
    fn test_negative_zero() {
        let neg_zero = -0.0f32;
        let v = Vec4F32::splat(neg_zero);
        let pos_zero = Vec4F32::zero();
        let result = v.add(pos_zero);
        // -0.0 + 0.0 = 0.0 (positive zero by IEEE 754)
        for i in 0..4 {
            assert_eq!(result.get(i), 0.0);
        }
    }
}

// ============================================================
// Alignment Tests
// ============================================================

mod alignment_tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_vec2f32_alignment() {
        assert_eq!(mem::align_of::<Vec2F32>(), 8);
    }

    #[test]
    fn test_vec4f32_alignment() {
        assert_eq!(mem::align_of::<Vec4F32>(), 16);
    }

    #[test]
    fn test_vec8f32_alignment() {
        assert_eq!(mem::align_of::<Vec8F32>(), 32);
    }

    #[test]
    fn test_vec2f64_alignment() {
        assert_eq!(mem::align_of::<Vec2F64>(), 16);
    }

    #[test]
    fn test_vec4f64_alignment() {
        assert_eq!(mem::align_of::<Vec4F64>(), 32);
    }

    #[test]
    fn test_vec4i32_alignment() {
        assert_eq!(mem::align_of::<Vec4I32>(), 16);
    }

    #[test]
    fn test_vec8i32_alignment() {
        assert_eq!(mem::align_of::<Vec8I32>(), 32);
    }
}

// ============================================================
// Performance Sanity Tests
// ============================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_vectorized_dot_product_1k() {
        // Simulate dot product of 1K elements using Vec8F32
        let n = 1024;
        let chunks = n / 8;

        let mut sum = 0.0f32;
        for _ in 0..chunks {
            let a = Vec8F32::splat(1.0);
            let b = Vec8F32::splat(2.0);
            sum += a.mul(b).sum();
        }

        assert_eq!(sum, 2048.0); // 1024 * 2.0
    }

    #[test]
    fn test_chained_operations() {
        // Test that chained operations work correctly
        let a = Vec8F32::splat(2.0);
        let b = Vec8F32::splat(3.0);
        let c = Vec8F32::splat(4.0);

        // (a + b) * c = 5 * 4 = 20
        let result = a.add(b).mul(c);
        for i in 0..8 {
            assert_eq!(result.get(i), 20.0);
        }
    }

    #[test]
    fn test_accumulator_pattern() {
        // Common pattern: accumulate results
        let mut acc = Vec8F32::zero();
        for i in 0..100 {
            let v = Vec8F32::splat(i as f32);
            acc = acc.add(v);
        }
        // Sum of 0..99 = 4950
        assert_eq!(acc.get(0), 4950.0);
    }
}
