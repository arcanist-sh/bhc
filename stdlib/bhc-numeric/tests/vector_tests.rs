//! Comprehensive Vector tests
//!
//! Tests for Vector<T> operations including:
//! - Construction and basic operations
//! - Arithmetic operations (add, sub, mul, div, dot)
//! - Slicing and concatenation
//! - SIMD-accelerated operations
//! - Property tests for algebraic laws

use bhc_numeric::vector::Vector;

// ============================================================
// Construction Tests
// ============================================================

mod construction_tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.len(), 4);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[3], 4.0);
    }

    #[test]
    fn test_zeros() {
        let v: Vector<f64> = Vector::zeros(5);
        assert_eq!(v.len(), 5);
        for i in 0..5 {
            assert_eq!(v[i], 0.0);
        }
    }

    #[test]
    fn test_fill() {
        let v = Vector::fill(4, 7.5f64);
        assert_eq!(v.len(), 4);
        for i in 0..4 {
            assert_eq!(v[i], 7.5);
        }
    }

    #[test]
    fn test_from_slice() {
        let data = [1.0, 2.0, 3.0];
        let v = Vector::from_slice(&data);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_empty_vector() {
        let v: Vector<f64> = Vector::from_vec(vec![]);
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_single_element() {
        let v = Vector::from_vec(vec![42.0f64]);
        assert_eq!(v.len(), 1);
        assert!(!v.is_empty());
        assert_eq!(v[0], 42.0);
    }
}

// ============================================================
// Access Tests
// ============================================================

mod access_tests {
    use super::*;

    #[test]
    fn test_index() {
        let v = Vector::from_vec(vec![10.0, 20.0, 30.0]);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[1], 20.0);
        assert_eq!(v[2], 30.0);
    }

    #[test]
    fn test_get() {
        let v = Vector::from_vec(vec![10.0, 20.0, 30.0]);
        assert_eq!(v.get(0), Some(&10.0));
        assert_eq!(v.get(2), Some(&30.0));
        assert_eq!(v.get(3), None);
    }

    #[test]
    fn test_get_mut() {
        let mut v = Vector::from_vec(vec![10.0, 20.0, 30.0]);
        if let Some(x) = v.get_mut(1) {
            *x = 25.0;
        }
        assert_eq!(v[1], 25.0);
    }

    #[test]
    fn test_as_slice() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let slice = v.as_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let slice = v.as_mut_slice();
        slice[0] = 10.0;
        assert_eq!(v[0], 10.0);
    }

    #[test]
    fn test_iter() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let sum: f64 = v.iter().sum();
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn test_iter_mut() {
        let mut v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        for x in v.iter_mut() {
            *x *= 2.0;
        }
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], 4.0);
        assert_eq!(v[2], 6.0);
    }
}

// ============================================================
// Slicing and Manipulation Tests
// ============================================================

mod slicing_tests {
    use super::*;

    #[test]
    fn test_slice() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let sliced = v.slice(1, 4).unwrap();
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced[0], 2.0);
        assert_eq!(sliced[1], 3.0);
        assert_eq!(sliced[2], 4.0);
    }

    #[test]
    fn test_slice_full() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let sliced = v.slice(0, 3).unwrap();
        assert_eq!(sliced.len(), 3);
    }

    #[test]
    fn test_slice_empty() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let sliced = v.slice(1, 1).unwrap();
        assert_eq!(sliced.len(), 0);
    }

    #[test]
    fn test_slice_invalid() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(v.slice(2, 1).is_none()); // start > end
        assert!(v.slice(0, 5).is_none()); // end > len
    }

    #[test]
    fn test_concat() {
        let v1 = Vector::from_vec(vec![1.0, 2.0]);
        let v2 = Vector::from_vec(vec![3.0, 4.0]);
        let result = v1.concat(&v2);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_concat_empty() {
        let v1 = Vector::from_vec(vec![1.0, 2.0]);
        let v2: Vector<f64> = Vector::from_vec(vec![]);
        let result = v1.concat(&v2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_reverse() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let rev = v.reverse();
        assert_eq!(rev[0], 4.0);
        assert_eq!(rev[1], 3.0);
        assert_eq!(rev[2], 2.0);
        assert_eq!(rev[3], 1.0);
    }

    #[test]
    fn test_reverse_single() {
        let v = Vector::from_vec(vec![42.0]);
        let rev = v.reverse();
        assert_eq!(rev[0], 42.0);
    }
}

// ============================================================
// Arithmetic Tests
// ============================================================

mod arithmetic_tests {
    use super::*;

    #[test]
    fn test_sum() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn test_sum_empty() {
        let v: Vector<f64> = Vector::zeros(0);
        assert_eq!(v.sum(), 0.0);
    }

    #[test]
    fn test_dot() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let dot = a.dot(&b).unwrap();
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_orthogonal() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 0.0]);
        let b: Vector<f64> = Vector::from_vec(vec![0.0, 1.0]);
        assert_eq!(a.dot(&b).unwrap(), 0.0);
    }

    #[test]
    fn test_dot_parallel() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 0.0]);
        let b: Vector<f64> = Vector::from_vec(vec![2.0, 0.0]);
        assert_eq!(a.dot(&b).unwrap(), 2.0);
    }

    #[test]
    fn test_add() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c[0], 5.0);
        assert_eq!(c[1], 7.0);
        assert_eq!(c[2], 9.0);
    }

    #[test]
    fn test_sub() {
        let a = Vector::from_vec(vec![10.0, 20.0, 30.0]);
        let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c[0], 9.0);
        assert_eq!(c[1], 18.0);
        assert_eq!(c[2], 27.0);
    }

    #[test]
    fn test_mul() {
        let a = Vector::from_vec(vec![2.0, 3.0, 4.0]);
        let b = Vector::from_vec(vec![5.0, 6.0, 7.0]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c[0], 10.0);
        assert_eq!(c[1], 18.0);
        assert_eq!(c[2], 28.0);
    }

    #[test]
    fn test_scale() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = v.scale(2.0);
        assert_eq!(scaled[0], 2.0);
        assert_eq!(scaled[1], 4.0);
        assert_eq!(scaled[2], 6.0);
    }

    #[test]
    fn test_scale_zero() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = v.scale(0.0);
        for i in 0..3 {
            assert_eq!(scaled[i], 0.0);
        }
    }

    #[test]
    fn test_negate() {
        let v = Vector::from_vec(vec![1.0, -2.0, 3.0]);
        let neg = v.negate();
        assert_eq!(neg[0], -1.0);
        assert_eq!(neg[1], 2.0);
        assert_eq!(neg[2], -3.0);
    }

    #[test]
    fn test_length_mismatch() {
        let a = Vector::from_vec(vec![1.0, 2.0]);
        let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(a.add(&b).is_none());
        assert!(a.sub(&b).is_none());
        assert!(a.mul(&b).is_none());
    }
}

// ============================================================
// Norm and Normalization Tests
// ============================================================

mod norm_tests {
    use super::*;

    #[test]
    fn test_norm() {
        let v: Vector<f64> = Vector::from_vec(vec![3.0, 4.0]);
        assert_eq!(v.norm(), 5.0); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_norm_unit() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 0.0, 0.0]);
        assert_eq!(v.norm(), 1.0);
    }

    #[test]
    fn test_norm_zero() {
        let v: Vector<f64> = Vector::zeros(3);
        assert_eq!(v.norm(), 0.0);
    }

    #[test]
    fn test_normalize() {
        let v: Vector<f64> = Vector::from_vec(vec![3.0, 4.0]);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-10);
        assert!((n[0] - 0.6).abs() < 1e-10);
        assert!((n[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_already_unit() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 0.0]);
        let n = v.normalize();
        assert_eq!(n[0], 1.0);
        assert_eq!(n[1], 0.0);
    }

    #[test]
    fn test_squared_norm() {
        let v: Vector<f64> = Vector::from_vec(vec![3.0, 4.0]);
        // squared_norm = dot(v, v) = 9 + 16 = 25
        assert_eq!(v.dot(&v).unwrap(), 25.0);
    }
}

// ============================================================
// Integer Vector Tests
// ============================================================

mod integer_tests {
    use super::*;

    #[test]
    fn test_i32_sum() {
        let v: Vector<i32> = Vector::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(v.sum(), 15);
    }

    #[test]
    fn test_i32_dot() {
        let a: Vector<i32> = Vector::from_vec(vec![1, 2, 3]);
        let b: Vector<i32> = Vector::from_vec(vec![4, 5, 6]);
        assert_eq!(a.dot(&b).unwrap(), 32);
    }

    #[test]
    fn test_i64_operations() {
        let v: Vector<i64> = Vector::from_vec(vec![1000000000, 2000000000, 3000000000]);
        assert_eq!(v.sum(), 6000000000);
    }

    #[test]
    fn test_negative_integers() {
        let v: Vector<i32> = Vector::from_vec(vec![-1, -2, 3, 4]);
        assert_eq!(v.sum(), 4);
    }
}

// ============================================================
// f32 Tests (Single Precision)
// ============================================================

mod f32_tests {
    use super::*;

    #[test]
    fn test_f32_sum() {
        let v: Vector<f32> = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn test_f32_dot() {
        let a: Vector<f32> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f32> = Vector::from_vec(vec![1.0, 1.0, 1.0]);
        assert_eq!(a.dot(&b).unwrap(), 6.0);
    }

    #[test]
    fn test_f32_precision() {
        // f32 has ~7 significant digits
        let v: Vector<f32> = Vector::from_vec(vec![1e7, 1.0, -1e7]);
        // Due to floating point, this might not be exactly 1.0
        let sum = v.sum();
        assert!((sum - 1.0).abs() < 1.0, "f32 precision test: sum = {}", sum);
    }
}

// ============================================================
// Large Vector Tests
// ============================================================

mod large_vector_tests {
    use super::*;

    #[test]
    fn test_large_sum() {
        let v: Vector<f64> = Vector::fill(10000, 1.0);
        assert_eq!(v.sum(), 10000.0);
    }

    #[test]
    fn test_large_dot() {
        let a: Vector<f64> = Vector::fill(10000, 1.0);
        let b: Vector<f64> = Vector::fill(10000, 2.0);
        assert_eq!(a.dot(&b).unwrap(), 20000.0);
    }

    #[test]
    fn test_1m_elements() {
        let n = 1_000_000;
        let v: Vector<f64> = Vector::fill(n, 1.0);
        assert_eq!(v.len(), n);
        assert_eq!(v.sum(), n as f64);
    }

    #[test]
    fn test_large_norm() {
        // Vector of all 1s: norm = sqrt(n)
        let n = 10000;
        let v: Vector<f64> = Vector::fill(n, 1.0);
        let expected = (n as f64).sqrt();
        assert!((v.norm() - expected).abs() < 1e-10);
    }
}

// ============================================================
// Clone and Conversion Tests
// ============================================================

mod conversion_tests {
    use super::*;

    #[test]
    fn test_clone() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let cloned = v.clone();
        assert_eq!(cloned[0], 1.0);
        assert_eq!(cloned[1], 2.0);
        assert_eq!(cloned[2], 3.0);
    }

    #[test]
    fn test_into_vec() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let raw: Vec<f64> = v.into_vec();
        assert_eq!(raw, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_clone_independence() {
        let mut v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let cloned = v.clone();
        v.as_mut_slice()[0] = 100.0;
        assert_eq!(cloned[0], 1.0); // Clone is independent
        assert_eq!(v[0], 100.0);
    }
}

// ============================================================
// Algebraic Property Tests
// ============================================================

mod property_tests {
    use super::*;

    #[test]
    fn test_add_commutative() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();
        for i in 0..3 {
            assert_eq!(ab[i], ba[i]);
        }
    }

    #[test]
    fn test_add_associative() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let c: Vector<f64> = Vector::from_vec(vec![7.0, 8.0, 9.0]);

        let ab_c = a.add(&b).unwrap().add(&c).unwrap();
        let a_bc = a.add(&b.add(&c).unwrap()).unwrap();

        for i in 0..3 {
            assert!((ab_c[i] - a_bc[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_add_identity() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let zero: Vector<f64> = Vector::zeros(3);
        let result = a.add(&zero).unwrap();
        for i in 0..3 {
            assert_eq!(result[i], a[i]);
        }
    }

    #[test]
    fn test_mul_commutative() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let ab = a.mul(&b).unwrap();
        let ba = b.mul(&a).unwrap();
        for i in 0..3 {
            assert_eq!(ab[i], ba[i]);
        }
    }

    #[test]
    fn test_dot_commutative() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b).unwrap(), b.dot(&a).unwrap());
    }

    #[test]
    fn test_scale_distributive() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let k: f64 = 2.0;

        // k * (a + b) = k*a + k*b
        let lhs = a.add(&b).unwrap().scale(k);
        let rhs = a.scale(k).add(&b.scale(k)).unwrap();

        for i in 0..3 {
            assert!((lhs[i] - rhs[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_negate_is_scale_minus_one() {
        let v = Vector::from_vec(vec![1.0, -2.0, 3.0]);
        let neg = v.negate();
        let scaled = v.scale(-1.0);
        for i in 0..3 {
            assert_eq!(neg[i], scaled[i]);
        }
    }

    #[test]
    fn test_double_negate() {
        let v = Vector::from_vec(vec![1.0, -2.0, 3.0]);
        let double_neg = v.negate().negate();
        for i in 0..3 {
            assert_eq!(v[i], double_neg[i]);
        }
    }

    #[test]
    fn test_cauchy_schwarz() {
        // |a . b| <= ||a|| * ||b||
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let dot_abs = a.dot(&b).unwrap().abs();
        let norm_product = a.norm() * b.norm();
        assert!(dot_abs <= norm_product + 1e-10);
    }

    #[test]
    fn test_triangle_inequality() {
        // ||a + b|| <= ||a|| + ||b||
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, -5.0, 6.0]);
        let sum_norm = a.add(&b).unwrap().norm();
        let norm_sum = a.norm() + b.norm();
        assert!(sum_norm <= norm_sum + 1e-10);
    }
}

// ============================================================
// Edge Cases
// ============================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_nan_handling() {
        let v = Vector::from_vec(vec![1.0, f64::NAN, 3.0]);
        assert!(v.sum().is_nan());
    }

    #[test]
    fn test_infinity_handling() {
        let v = Vector::from_vec(vec![1.0, f64::INFINITY, 3.0]);
        assert!(v.sum().is_infinite());
    }

    #[test]
    fn test_very_small_values() {
        let tiny = f64::MIN_POSITIVE;
        let v = Vector::from_vec(vec![tiny, tiny, tiny]);
        assert!(v.sum() > 0.0);
    }

    #[test]
    fn test_very_large_values() {
        let big = f64::MAX / 4.0;
        let v = Vector::from_vec(vec![big, big]);
        // Should not overflow
        assert!(!v.sum().is_infinite());
    }

    #[test]
    fn test_mixed_signs_cancellation() {
        let v: Vector<f64> = Vector::from_vec(vec![1e16, 1.0, -1e16]);
        // Due to floating point, 1.0 may be lost
        let sum: f64 = v.sum();
        // Just verify it doesn't crash and returns something finite
        assert!(!sum.is_nan());
    }
}
