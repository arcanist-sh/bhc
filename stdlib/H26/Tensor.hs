-- |
-- Module      : H26.Tensor
-- Description : Tensor operations for numeric computing
-- License     : BSD-3-Clause
--
-- The H26.Tensor module provides tensor operations required for
-- H26-Numeric conformance. All operations follow the Tensor IR
-- lowering contract specified in H26-SPEC Section 7.

{-# HASKELL_EDITION 2026 #-}
{-# PROFILE Numeric #-}

module H26.Tensor
  ( -- * Tensor Type
    Tensor
  , DType(..)
  , Shape
  , Stride

    -- * Construction
  , zeros
  , ones
  , fromList
  , fromUArray
  , generate

    -- * Properties
  , shape
  , rank
  , size
  , dtype
  , strides
  , isContiguous

    -- * Views (no copy)
  , reshape
  , slice
  , transpose
  , view

    -- * Elementwise Operations
  , map
  , zipWith
  , zipWith3

    -- * Reductions
  , sum
  , product
  , mean
  , max
  , min
  , argmax
  , argmin
  , fold
  , foldl'

    -- * Linear Algebra
  , dot
  , matmul
  , outer
  , trace
  , diag

    -- * Parallel Operations
  , parMap
  , parReduce
  , parFor

    -- * Control
  , materialize
  , forceEval

    -- * Re-exports for common dtypes
  , F32
  , F64
  , I32
  , I64
  ) where

import Prelude hiding (map, zipWith, sum, product, max, min)

-- | The tensor data type.
--
-- Tensors are multidimensional arrays with shape and stride information.
-- Views share underlying data without copying.
data Tensor (dtype :: DType)

-- | Element type for tensors.
data DType
  = F32   -- ^ 32-bit float
  | F64   -- ^ 64-bit float
  | I32   -- ^ 32-bit signed integer
  | I64   -- ^ 64-bit signed integer
  | U32   -- ^ 32-bit unsigned integer
  | U64   -- ^ 64-bit unsigned integer
  | Bool  -- ^ Boolean

-- | Shape is a list of dimension sizes.
type Shape = [Int]

-- | Strides in bytes for each dimension.
type Stride = [Int]

-- Type aliases for convenience
type F32 = Float
type F64 = Double
type I32 = Int
type I64 = Integer

-- | Create a tensor filled with zeros.
--
-- @zeros [2, 3]@ creates a 2x3 matrix of zeros.
zeros :: Shape -> Tensor F32

-- | Create a tensor filled with ones.
ones :: Shape -> Tensor F32

-- | Create a tensor from a list.
--
-- @fromList [2, 3] [1..6]@ creates a 2x3 matrix.
fromList :: Shape -> [a] -> Tensor a

-- | Create a tensor from an unboxed array.
fromUArray :: Shape -> UArray a -> Tensor a

-- | Generate a tensor using a function.
generate :: Shape -> (Int -> a) -> Tensor a

-- | Get the shape of a tensor.
shape :: Tensor a -> Shape

-- | Get the rank (number of dimensions).
rank :: Tensor a -> Int

-- | Get the total number of elements.
size :: Tensor a -> Int

-- | Get the element type.
dtype :: Tensor a -> DType

-- | Get the strides.
strides :: Tensor a -> Stride

-- | Check if the tensor is contiguous in memory.
isContiguous :: Tensor a -> Bool

-- | Reshape a tensor (view, no copy if possible).
--
-- The new shape must have the same total number of elements.
reshape :: Shape -> Tensor a -> Tensor a

-- | Slice a tensor (view, no copy).
slice :: [(Int, Int)] -> Tensor a -> Tensor a

-- | Transpose a tensor (view, no copy).
--
-- @transpose [1, 0] m@ transposes a matrix.
transpose :: [Int] -> Tensor a -> Tensor a

-- | Create a view with explicit shape and strides.
view :: Shape -> Stride -> Tensor a -> Tensor a

-- | Map a function over tensor elements.
--
-- GUARANTEED to fuse with other maps (H26-SPEC 8.1).
map :: (a -> b) -> Tensor a -> Tensor b

-- | Zip two tensors with a function.
--
-- GUARANTEED to fuse with maps (H26-SPEC 8.1).
zipWith :: (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c

-- | Zip three tensors with a function.
zipWith3 :: (a -> b -> c -> d) -> Tensor a -> Tensor b -> Tensor c -> Tensor d

-- | Sum all elements.
--
-- GUARANTEED to fuse with maps (H26-SPEC 8.1).
sum :: Num a => Tensor a -> a

-- | Product of all elements.
product :: Num a => Tensor a -> a

-- | Mean of all elements.
mean :: Fractional a => Tensor a -> a

-- | Maximum element.
max :: Ord a => Tensor a -> a

-- | Minimum element.
min :: Ord a => Tensor a -> a

-- | Index of maximum element.
argmax :: Ord a => Tensor a -> Int

-- | Index of minimum element.
argmin :: Ord a => Tensor a -> Int

-- | Fold over tensor elements.
fold :: (b -> a -> b) -> b -> Tensor a -> b

-- | Strict left fold.
foldl' :: (b -> a -> b) -> b -> Tensor a -> b

-- | Dot product of two 1D tensors.
dot :: Num a => Tensor a -> Tensor a -> a

-- | Matrix multiplication.
matmul :: Num a => Tensor a -> Tensor a -> Tensor a

-- | Outer product of two 1D tensors.
outer :: Num a => Tensor a -> Tensor a -> Tensor a

-- | Trace of a matrix.
trace :: Num a => Tensor a -> a

-- | Diagonal of a matrix.
diag :: Tensor a -> Tensor a

-- | Parallel map.
parMap :: (a -> b) -> Tensor a -> Tensor b

-- | Parallel reduction.
parReduce :: Monoid m => (a -> m) -> Tensor a -> m

-- | Parallel for loop.
parFor :: (Int, Int) -> (Int -> ()) -> ()

-- | Force materialization (no fusion).
materialize :: Tensor a -> Tensor a

-- | Force evaluation of tensor elements.
forceEval :: Tensor a -> Tensor a
