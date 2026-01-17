-- Benchmark: saxpy
-- Category: numeric
-- Profile: numeric
-- Spec: H26-SPEC Section 8 (Fusion Guarantees)
--
-- SAXPY: y = a*x + y (Single-precision A*X Plus Y)
-- This is a fundamental BLAS Level 1 operation.

{-# PROFILE Numeric #-}

module SaxpyBench where

import H26.Tensor
import H26.BLAS

-- | Naive saxpy using map and zipWith
-- MUST fuse to single traversal per H26-SPEC Section 8.1
saxpyNaive :: Float -> Tensor Float -> Tensor Float -> Tensor Float
saxpyNaive a x y = zipWith (+) (map (*a) x) y

-- | Using BLAS axpy operation
saxpyBLAS :: Float -> Tensor Float -> Tensor Float -> Tensor Float
saxpyBLAS a x y = axpy a (toList x) (toList y)
  & fromList

-- | Point-free style
saxpyPointFree :: Float -> Tensor Float -> Tensor Float -> Tensor Float
saxpyPointFree a = zipWith (+) . map (*a)

-- | In-place version (returns modified y)
saxpyInPlace :: Float -> Tensor Float -> Tensor Float -> IO (Tensor Float)
saxpyInPlace a x y = do
  y' <- thaw y
  axpyInPlace a (toList x) y'
  freeze y'

-- | Vector sizes for benchmarking
sizes :: [Int]
sizes = [1000, 10000, 100000, 1000000]

-- | Expected performance characteristics:
-- - O(n) time complexity
-- - O(1) allocation for fused version
-- - O(n) allocation for in-place (output only)
-- - SIMD vectorization (4-8x speedup on AVX/AVX2)
-- - Memory bandwidth bound for large vectors

-- | Validation test
validate :: Bool
validate =
  let a = 2.0
      x = fromList [1, 2, 3, 4, 5]
      y = fromList [10, 20, 30, 40, 50]
      -- Expected: [2*1+10, 2*2+20, 2*3+30, 2*4+40, 2*5+50]
      --         = [12, 24, 36, 48, 60]
      expected = fromList [12, 24, 36, 48, 60]
  in saxpyNaive a x y == expected

-- | Double-precision variant (DAXPY)
daxpyNaive :: Double -> Tensor Double -> Tensor Double -> Tensor Double
daxpyNaive a x y = zipWith (+) (map (*a) x) y

-- | FMA optimization test
-- This should use fused multiply-add instructions when available
saxpyFMA :: Float -> Tensor Float -> Tensor Float -> Tensor Float
saxpyFMA a x y = zipWith (fma a) x y
  where
    fma :: Float -> Float -> Float -> Float
    fma a' x' y' = a' * x' + y'  -- Compiler should use FMA

-- | Expected results (reference, machine-dependent):
-- Size     | Time (μs) | Memory BW
-- ---------|-----------|------------
-- 1K       | ~0.3      | ~10 GB/s
-- 10K      | ~3        | ~10 GB/s
-- 100K     | ~30       | ~10 GB/s
-- 1M       | ~300      | ~10 GB/s
--
-- Note: Memory BW = 3*n*4 bytes / time (read x, read y, write y)

-- Helper
(&) :: a -> (a -> b) -> b
x & f = f x
