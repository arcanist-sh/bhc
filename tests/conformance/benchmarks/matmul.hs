-- Benchmark: matrix-multiply
-- Category: numeric
-- Profile: numeric
-- Spec: H26-SPEC Section 7 (Tensor Model), Section 8 (Fusion)
--
-- Matrix multiplication: C = A × B
-- This is the fundamental BLAS Level 3 operation.

{-# PROFILE Numeric #-}

module MatmulBench where

import H26.Tensor
import H26.BLAS

-- | Naive matrix multiplication using tensor operations
-- Should be optimized by the compiler but may not match BLAS
matmulNaive :: Tensor Float -> Tensor Float -> Tensor Float
matmulNaive a b =
  let [m, k1] = shape a
      [k2, n] = shape b
  in generate [m, n] $ \[i, j] ->
       sum [ a ! [i, l] * b ! [l, j] | l <- [0..k1-1] ]

-- | Using BLAS GEMM (C = alpha*A*B + beta*C)
matmulBLAS :: Tensor Float -> Tensor Float -> Tensor Float
matmulBLAS a b = matmul a b

-- | Using explicit GEMM parameters
matmulGEMM :: Tensor Float -> Tensor Float -> Tensor Float
matmulGEMM a b =
  let c = zeros (rows a) (cols b)
  in gemm NoTrans NoTrans 1.0 a b 0.0 c

-- | Transposed multiply: A^T × B
matmulTransA :: Tensor Float -> Tensor Float -> Tensor Float
matmulTransA a b =
  gemm Trans NoTrans 1.0 a b 0.0 (zeros (cols a) (cols b))

-- | Matrix sizes for benchmarking
sizes :: [(Int, Int, Int)]  -- (M, N, K)
sizes =
  [ (64, 64, 64)
  , (128, 128, 128)
  , (256, 256, 256)
  , (512, 512, 512)
  , (1024, 1024, 1024)
  ]

-- | Expected performance characteristics:
-- - O(n³) time complexity for n×n matrices
-- - O(n²) memory for output
-- - Achieved performance depends heavily on:
--   - Cache blocking (L1/L2/L3 hierarchy)
--   - SIMD vectorization
--   - Memory layout (row-major vs column-major)
--   - BLAS implementation quality

-- | Validation test
validate :: Bool
validate =
  let a = fromRows [[1, 2], [3, 4]]
      b = fromRows [[5, 6], [7, 8]]
      -- Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
      --         = [[19, 22], [43, 50]]
      expected = fromRows [[19, 22], [43, 50]]
  in matmulBLAS a b == expected

-- | Double-precision variant (DGEMM)
dgemm :: Tensor Double -> Tensor Double -> Tensor Double
dgemm = matmul

-- | Batched matrix multiply
batchedMatmul :: [Tensor Float] -> [Tensor Float] -> [Tensor Float]
batchedMatmul as bs = gemmBatched NoTrans NoTrans 1.0 as bs 0.0 cs
  where
    cs = [ zeros (rows a) (cols b) | (a, b) <- zip as bs ]

-- | Expected results (reference, machine-dependent):
-- Size     | Time (ms) | GFLOPS
-- ---------|-----------|------------
-- 64³      | ~0.05     | ~10 GFLOPS
-- 128³     | ~0.3      | ~15 GFLOPS
-- 256³     | ~2        | ~20 GFLOPS
-- 512³     | ~15       | ~25 GFLOPS
-- 1024³    | ~100      | ~25 GFLOPS
--
-- Note: FLOPS = 2*M*N*K / time
-- Peak performance depends on CPU (e.g., ~200 GFLOPS for modern x86)
-- Typical achieved: 50-80% of peak with optimized BLAS

-- | Memory layout test
-- Row-major vs column-major can affect performance significantly
layoutTest :: Tensor Float -> Tensor Float -> (Tensor Float, Tensor Float)
layoutTest a b =
  let rowMajor = matmul (asRowMajor a) (asRowMajor b)
      colMajor = matmul (asColMajor a) (asColMajor b)
  in (rowMajor, colMajor)

-- | Strassen algorithm threshold test
-- For large matrices, Strassen's algorithm may be faster
-- but has different numerical properties
strassenThreshold :: Int
strassenThreshold = 512  -- Typical crossover point

-- Helper functions
rows :: Tensor a -> Int
rows t = head (shape t)

cols :: Tensor a -> Int
cols t = shape t !! 1

zeros :: Int -> Int -> Tensor Float
zeros m n = generate [m, n] (const 0)

fromRows :: [[Float]] -> Tensor Float
fromRows = undefined

shape :: Tensor a -> [Int]
shape = undefined

generate :: [Int] -> ([Int] -> a) -> Tensor a
generate = undefined

(!) :: Tensor a -> [Int] -> a
(!) = undefined
