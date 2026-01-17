-- Benchmark: dot-product
-- Category: numeric
-- Profile: numeric
-- Spec: H26-SPEC Section 8 (Fusion Guarantees)
--
-- This benchmark tests the fundamental dot product operation.
-- It MUST fuse without intermediate allocation.

{-# PROFILE Numeric #-}

module DotProductBench where

import H26.Tensor
import H26.BLAS

-- | Naive dot product using sum and zipWith
-- MUST fuse to single traversal per H26-SPEC Section 8.1
dotNaive :: Tensor Float -> Tensor Float -> Float
dotNaive xs ys = sum (zipWith (*) xs ys)

-- | Using BLAS dot operation
dotBLAS :: Tensor Float -> Tensor Float -> Float
dotBLAS = vdot

-- | Manual fold (should be optimized equivalently)
dotFold :: Tensor Float -> Tensor Float -> Float
dotFold xs ys = foldl' (+) 0 (zipWith (*) xs ys)

-- | Vector sizes for benchmarking
sizes :: [Int]
sizes = [1000, 10000, 100000, 1000000]

-- | Expected performance characteristics:
-- - O(n) time complexity
-- - O(1) allocation (no intermediate arrays)
-- - SIMD vectorization on supported hardware
-- - Memory bandwidth bound for large vectors

-- | Reference implementation (pure Haskell)
dotRef :: [Float] -> [Float] -> Float
dotRef xs ys = go 0 xs ys
  where
    go !acc [] _ = acc
    go !acc _ [] = acc
    go !acc (x:xs') (y:ys') = go (acc + x*y) xs' ys'

-- | Validation test
validate :: Bool
validate =
  let xs = fromList [1, 2, 3, 4, 5]
      ys = fromList [2, 3, 4, 5, 6]
      expected = 1*2 + 2*3 + 3*4 + 4*5 + 5*6  -- 70
  in dotNaive xs ys == expected
     && dotBLAS xs ys == expected
     && dotFold xs ys == expected

-- | Benchmark configuration
-- Run each benchmark with:
--   bhc bench --iterations 1000 --warmup 100

-- | Expected results (reference, machine-dependent):
-- Size     | Time (μs) | FLOPS
-- ---------|-----------|------------
-- 1K       | ~0.5      | ~4 GFLOPS
-- 10K      | ~5        | ~4 GFLOPS
-- 100K     | ~50       | ~4 GFLOPS
-- 1M       | ~500      | ~4 GFLOPS
--
-- Note: FLOPS = 2*n / time (mul + add per element)
