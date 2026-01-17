-- Benchmark: reductions
-- Category: numeric
-- Profile: numeric
-- Spec: H26-SPEC Section 8 (Fusion Guarantees)
--
-- Reduction operations: sum, product, min, max, mean, etc.
-- These MUST fuse with preceding map operations.

{-# PROFILE Numeric #-}

module ReductionsBench where

import H26.Tensor
import H26.Numeric

-- | Basic sum
-- Must be O(n) with O(1) allocation
sumBasic :: Tensor Float -> Float
sumBasic = sum

-- | Sum with preceding map (MUST fuse)
sumMap :: Tensor Float -> Float
sumMap xs = sum (map (*2) xs)

-- | Product
productBasic :: Tensor Float -> Float
productBasic = product

-- | Product with map (MUST fuse)
productMap :: Tensor Float -> Float
productMap xs = product (map (\x -> x + 1) xs)

-- | Minimum
minBasic :: Tensor Float -> Float
minBasic = minimum

-- | Maximum
maxBasic :: Tensor Float -> Float
maxBasic = maximum

-- | Argmin (index of minimum)
argmin :: Tensor Float -> Int
argmin = minIndex

-- | Argmax (index of maximum)
argmax :: Tensor Float -> Int
argmax = maxIndex

-- | Mean (average)
-- Should be single-pass for efficiency
meanBasic :: Tensor Float -> Float
meanBasic xs = sum xs / fromIntegral (length xs)

-- | Variance (sample variance)
-- Requires two passes or online algorithm
varianceBasic :: Tensor Float -> Float
varianceBasic = variance

-- | Standard deviation
stddevBasic :: Tensor Float -> Float
stddevBasic = stddev

-- | Norm calculations
normL1 :: Tensor Float -> Float
normL1 = vnorm1 . toList

normL2 :: Tensor Float -> Float
normL2 = vnorm . toList

normInf :: Tensor Float -> Float
normInf = vnormInf . toList

-- | Kahan summation (for better numerical accuracy)
sumKahan :: Tensor Float -> Float
sumKahan xs = kahanSum (toList xs)

-- | Pairwise summation (O(n log n) accuracy)
sumPairwise :: Tensor Float -> Float
sumPairwise xs = pairwiseSum (toList xs)

-- | Parallel reduction
sumParallel :: Tensor Float -> Float
sumParallel xs = sum xs  -- Automatically parallelized in Numeric profile

-- | Reduction along axis
sumAxis0 :: Tensor Float -> Tensor Float
sumAxis0 = reduceSum 0

sumAxis1 :: Tensor Float -> Tensor Float
sumAxis1 = reduceSum 1

-- | Vector sizes for benchmarking
sizes :: [Int]
sizes = [1000, 10000, 100000, 1000000, 10000000]

-- | Validation tests
validateSum :: Bool
validateSum =
  let xs = fromList [1, 2, 3, 4, 5]
  in sumBasic xs == 15

validateMean :: Bool
validateMean =
  let xs = fromList [1, 2, 3, 4, 5]
  in abs (meanBasic xs - 3.0) < 1e-6

-- | Expected performance characteristics:
-- - O(n) time complexity
-- - O(1) allocation
-- - Memory bandwidth limited for large arrays
-- - Parallel reduction for multi-core

-- | Expected results (reference, machine-dependent):
-- Size     | Time (μs) | Memory BW
-- ---------|-----------|------------
-- 1K       | ~0.2      | ~20 GB/s
-- 10K      | ~2        | ~20 GB/s
-- 100K     | ~20       | ~20 GB/s
-- 1M       | ~200      | ~20 GB/s
-- 10M      | ~2000     | ~20 GB/s
--
-- Note: Memory BW = n * 4 bytes / time

-- | Numerical stability test
-- Large sums can accumulate floating-point error
stabilityTest :: Bool
stabilityTest =
  let n = 10000000
      xs = replicate n 1.0
      naiveSum = sum (fromList xs)
      kahanSum' = kahanSum xs
      -- Kahan should be more accurate
  in abs (kahanSum' - fromIntegral n) < abs (naiveSum - fromIntegral n)

-- Helper functions
length :: Tensor a -> Int
length = undefined

reduceSum :: Int -> Tensor Float -> Tensor Float
reduceSum = undefined

replicate :: Int -> a -> [a]
replicate n x = take n (repeat x)
  where
    repeat x' = x' : repeat x'

take :: Int -> [a] -> [a]
take 0 _ = []
take _ [] = []
take n (x:xs) = x : take (n-1) xs
