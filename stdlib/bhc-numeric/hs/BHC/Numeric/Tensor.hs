-- |
-- Module      : BHC.Numeric.Tensor
-- Description : Shape-indexed tensors with guaranteed fusion
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- N-dimensional arrays with compile-time or runtime shape checking.
-- All operations are designed for fusion - see BHC Spec Section 8.

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}

module BHC.Numeric.Tensor (
    -- * Tensor type
    Tensor,
    DType(..),
    Shape,

    -- * Construction
    zeros, ones, full,
    fromList, fromListFlat,
    arange, linspace,
    eye, diag,
    rand, randn,

    -- * Shape operations
    shape, rank, size,
    reshape, flatten, squeeze, unsqueeze,
    transpose, permute,
    broadcast, expand,

    -- * Indexing
    (!), index,
    slice, narrow,
    gather, scatter,
    masked,

    -- * Element-wise operations (MUST FUSE)
    tMap, tZipWith,
    tAdd, tSub, tMul, tDiv,
    tNeg, tAbs, tSign,
    tSqrt, tExp, tLog, tPow,
    tSin, tCos, tTan,
    tSinh, tCosh, tTanh,
    tFloor, tCeil, tRound,
    tMin, tMax, tClamp,

    -- * Reductions (MUST FUSE with maps)
    tSum, tProduct, tMean, tVar, tStd,
    tMin', tMax', tArgmin, tArgmax,
    tAny, tAll,

    -- * Reduction along axis
    sumAxis, productAxis, meanAxis,
    minAxis, maxAxis,
    argminAxis, argmaxAxis,

    -- * Linear algebra
    dot, matmul, (@),
    outer, inner,
    norm, normalize,
    trace, det, inv,
    solve, lstsq,
    eig, svd, qr, cholesky,

    -- * Comparison
    tEq, tNe, tLt, tLe, tGt, tGe,
    tIsNan, tIsInf, tIsFinite,

    -- * Type casting
    cast, asType,

    -- * Utilities
    clone, contiguous,
    toList, toLists,
    materialize,

    -- * Arena allocation
    withTensorArena,
) where

import BHC.Prelude

-- ============================================================
-- Types
-- ============================================================

-- | N-dimensional tensor with element type @a@.
--
-- Tensors track their shape, strides, and memory layout for
-- efficient fusion and vectorization.
data Tensor a = Tensor
    { tensorData   :: !TensorData
    , tensorShape  :: ![Int]
    , tensorStride :: ![Int]
    , tensorOffset :: !Int
    }

-- | Internal tensor data (opaque).
data TensorData

-- | Data type enumeration.
data DType
    = Float16
    | Float32
    | Float64
    | Int8
    | Int16
    | Int32
    | Int64
    | UInt8
    | Bool
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Shape type (list of dimensions).
type Shape = [Int]

-- ============================================================
-- Construction
-- ============================================================

-- | Create tensor filled with zeros.
foreign import ccall "bhc_tensor_zeros"
    zeros :: Shape -> IO (Tensor Float)

-- | Create tensor filled with ones.
foreign import ccall "bhc_tensor_ones"
    ones :: Shape -> IO (Tensor Float)

-- | Create tensor filled with a constant value.
full :: Shape -> a -> IO (Tensor a)
full sh val = undefined

-- | Create tensor from nested list.
--
-- >>> fromList [[1,2,3], [4,5,6]]
-- Tensor [2, 3] ...
fromList :: [[a]] -> Tensor a
fromList = undefined

-- | Create tensor from flat list with shape.
--
-- >>> fromListFlat [2, 3] [1,2,3,4,5,6]
-- Tensor [2, 3] ...
fromListFlat :: Shape -> [a] -> Tensor a
fromListFlat = undefined

-- | Create 1D tensor with range [start, end).
--
-- >>> arange 0 10 1
-- Tensor [10] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
arange :: Num a => a -> a -> a -> Tensor a
arange start end step = undefined

-- | Create 1D tensor with evenly spaced values.
--
-- >>> linspace 0 1 5
-- Tensor [5] [0.0, 0.25, 0.5, 0.75, 1.0]
linspace :: Fractional a => a -> a -> Int -> Tensor a
linspace start end n = undefined

-- | Create identity matrix.
--
-- >>> eye 3
-- Tensor [3, 3] [[1,0,0], [0,1,0], [0,0,1]]
eye :: Num a => Int -> Tensor a
eye n = undefined

-- | Create diagonal matrix from vector.
diag :: Tensor a -> Tensor a
diag = undefined

-- | Create tensor with uniform random values in [0, 1).
rand :: Shape -> IO (Tensor Float)
rand = undefined

-- | Create tensor with standard normal random values.
randn :: Shape -> IO (Tensor Float)
randn = undefined

-- ============================================================
-- Shape Operations
-- ============================================================

-- | Get tensor shape.
shape :: Tensor a -> Shape
shape = tensorShape

-- | Get tensor rank (number of dimensions).
rank :: Tensor a -> Int
rank = length . tensorShape

-- | Get total number of elements.
size :: Tensor a -> Int
size = product . tensorShape

-- | Reshape tensor to new shape.
--
-- The new shape must have the same number of elements.
--
-- >>> reshape [6] (zeros [2, 3])
-- Tensor [6] ...
--
-- >>> reshape [3, 2] (zeros [2, 3])
-- Tensor [3, 2] ...
reshape :: Shape -> Tensor a -> Tensor a
reshape newShape t
    | product newShape /= size t = error "reshape: size mismatch"
    | otherwise = t { tensorShape = newShape, tensorStride = computeStrides newShape }

-- | Flatten tensor to 1D.
flatten :: Tensor a -> Tensor a
flatten t = reshape [size t] t

-- | Remove dimensions of size 1.
squeeze :: Tensor a -> Tensor a
squeeze t = t { tensorShape = filter (/= 1) (tensorShape t) }

-- | Add dimension of size 1 at position.
unsqueeze :: Int -> Tensor a -> Tensor a
unsqueeze dim t = t { tensorShape = insertAt dim 1 (tensorShape t) }

-- | Transpose a 2D tensor.
transpose :: Tensor a -> Tensor a
transpose t = permute [1, 0] t

-- | Permute dimensions.
permute :: [Int] -> Tensor a -> Tensor a
permute perm t = t
    { tensorShape = permuteList perm (tensorShape t)
    , tensorStride = permuteList perm (tensorStride t)
    }

-- | Broadcast tensor to larger shape.
broadcast :: Shape -> Tensor a -> Tensor a
broadcast targetShape t = undefined

-- | Expand tensor (alias for broadcast).
expand :: Shape -> Tensor a -> Tensor a
expand = broadcast

-- ============================================================
-- Indexing
-- ============================================================

-- | Index into tensor.
--
-- >>> t ! [0, 1, 2]
-- 3.14
(!) :: Tensor a -> [Int] -> a
(!) = index

-- | Index into tensor (function form).
index :: Tensor a -> [Int] -> a
index t indices = undefined

-- | Slice tensor along dimensions.
--
-- >>> slice [(0, 2), (1, 3)] t
-- Tensor [2, 2] ...
slice :: [(Int, Int)] -> Tensor a -> Tensor a
slice ranges t = undefined

-- | Narrow tensor along one dimension.
narrow :: Int -> Int -> Int -> Tensor a -> Tensor a
narrow dim start len t = undefined

-- | Gather elements along axis using indices.
gather :: Int -> Tensor Int -> Tensor a -> Tensor a
gather axis indices t = undefined

-- | Scatter values into tensor at indices.
scatter :: Int -> Tensor Int -> Tensor a -> Tensor a -> Tensor a
scatter axis indices src dst = undefined

-- | Select elements using boolean mask.
masked :: Tensor Bool -> Tensor a -> Tensor a
masked mask t = undefined

-- ============================================================
-- Element-wise Operations (MUST FUSE)
-- ============================================================

-- | Map function over tensor elements.
--
-- ==== __Fusion__
--
-- @tMap f (tMap g t)@ MUST fuse to @tMap (f . g) t@.
{-# RULES
"tMap/tMap" forall f g t. tMap f (tMap g t) = tMap (f . g) t
#-}
tMap :: (a -> b) -> Tensor a -> Tensor b
tMap f t = undefined

-- | Zip two tensors with a binary function.
--
-- ==== __Fusion__
--
-- @tZipWith f (tMap g a) (tMap h b)@ MUST fuse to single traversal.
{-# RULES
"tZipWith/tMap/tMap" forall f g h a b.
    tZipWith f (tMap g a) (tMap h b) = tZipWith (\x y -> f (g x) (h y)) a b
#-}
tZipWith :: (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
tZipWith f t1 t2 = undefined

-- | Element-wise addition.
tAdd :: Num a => Tensor a -> Tensor a -> Tensor a
tAdd = tZipWith (+)

-- | Element-wise subtraction.
tSub :: Num a => Tensor a -> Tensor a -> Tensor a
tSub = tZipWith (-)

-- | Element-wise multiplication.
tMul :: Num a => Tensor a -> Tensor a -> Tensor a
tMul = tZipWith (*)

-- | Element-wise division.
tDiv :: Fractional a => Tensor a -> Tensor a -> Tensor a
tDiv = tZipWith (/)

-- | Negate all elements.
tNeg :: Num a => Tensor a -> Tensor a
tNeg = tMap negate

-- | Absolute value.
tAbs :: Num a => Tensor a -> Tensor a
tAbs = tMap abs

-- | Sign function.
tSign :: Num a => Tensor a -> Tensor a
tSign = tMap signum

-- | Square root.
foreign import ccall "bhc_tensor_sqrt"
    tSqrt :: Tensor Float -> Tensor Float

-- | Exponential.
foreign import ccall "bhc_tensor_exp"
    tExp :: Tensor Float -> Tensor Float

-- | Natural logarithm.
foreign import ccall "bhc_tensor_log"
    tLog :: Tensor Float -> Tensor Float

-- | Power.
tPow :: Floating a => Tensor a -> Tensor a -> Tensor a
tPow = tZipWith (**)

-- | Sine.
foreign import ccall "bhc_tensor_sin"
    tSin :: Tensor Float -> Tensor Float

-- | Cosine.
foreign import ccall "bhc_tensor_cos"
    tCos :: Tensor Float -> Tensor Float

-- | Tangent.
foreign import ccall "bhc_tensor_tan"
    tTan :: Tensor Float -> Tensor Float

-- | Hyperbolic sine.
tSinh :: Floating a => Tensor a -> Tensor a
tSinh = tMap sinh

-- | Hyperbolic cosine.
tCosh :: Floating a => Tensor a -> Tensor a
tCosh = tMap cosh

-- | Hyperbolic tangent.
foreign import ccall "bhc_tensor_tanh"
    tTanh :: Tensor Float -> Tensor Float

-- | Floor.
tFloor :: RealFrac a => Tensor a -> Tensor a
tFloor = tMap (fromIntegral . floor)

-- | Ceiling.
tCeil :: RealFrac a => Tensor a -> Tensor a
tCeil = tMap (fromIntegral . ceiling)

-- | Round.
tRound :: RealFrac a => Tensor a -> Tensor a
tRound = tMap (fromIntegral . round)

-- | Element-wise minimum.
tMin :: Ord a => Tensor a -> Tensor a -> Tensor a
tMin = tZipWith min

-- | Element-wise maximum.
tMax :: Ord a => Tensor a -> Tensor a -> Tensor a
tMax = tZipWith max

-- | Clamp values to range.
tClamp :: Ord a => a -> a -> Tensor a -> Tensor a
tClamp lo hi = tMap (max lo . min hi)

-- ============================================================
-- Reductions (MUST FUSE with maps)
-- ============================================================

-- | Sum of all elements.
--
-- ==== __Fusion__
--
-- @tSum (tMap f t)@ MUST fuse to single traversal.
{-# RULES
"tSum/tMap" forall f t. tSum (tMap f t) = tFoldl' (\acc x -> acc + f x) 0 t
#-}
foreign import ccall "bhc_tensor_sum"
    tSum :: Num a => Tensor a -> a

-- | Product of all elements.
{-# RULES
"tProduct/tMap" forall f t. tProduct (tMap f t) = tFoldl' (\acc x -> acc * f x) 1 t
#-}
tProduct :: Num a => Tensor a -> a
tProduct = undefined

-- | Mean of all elements.
tMean :: Fractional a => Tensor a -> a
tMean t = tSum t / fromIntegral (size t)

-- | Variance of all elements.
tVar :: Floating a => Tensor a -> a
tVar t =
    let m = tMean t
        n = fromIntegral (size t)
    in tSum (tMap (\x -> (x - m) ^ 2) t) / n

-- | Standard deviation.
tStd :: Floating a => Tensor a -> a
tStd = sqrt . tVar

-- | Minimum element.
tMin' :: Ord a => Tensor a -> a
tMin' = undefined

-- | Maximum element.
tMax' :: Ord a => Tensor a -> a
tMax' = undefined

-- | Index of minimum element.
tArgmin :: Ord a => Tensor a -> Int
tArgmin = undefined

-- | Index of maximum element.
tArgmax :: Ord a => Tensor a -> Int
tArgmax = undefined

-- | True if any element is True.
tAny :: Tensor Bool -> Bool
tAny = undefined

-- | True if all elements are True.
tAll :: Tensor Bool -> Bool
tAll = undefined

-- Internal fold
tFoldl' :: (b -> a -> b) -> b -> Tensor a -> b
tFoldl' = undefined

-- ============================================================
-- Reduction Along Axis
-- ============================================================

-- | Sum along axis.
sumAxis :: Num a => Int -> Tensor a -> Tensor a
sumAxis = undefined

-- | Product along axis.
productAxis :: Num a => Int -> Tensor a -> Tensor a
productAxis = undefined

-- | Mean along axis.
meanAxis :: Fractional a => Int -> Tensor a -> Tensor a
meanAxis = undefined

-- | Minimum along axis.
minAxis :: Ord a => Int -> Tensor a -> Tensor a
minAxis = undefined

-- | Maximum along axis.
maxAxis :: Ord a => Int -> Tensor a -> Tensor a
maxAxis = undefined

-- | Index of minimum along axis.
argminAxis :: Ord a => Int -> Tensor a -> Tensor Int
argminAxis = undefined

-- | Index of maximum along axis.
argmaxAxis :: Ord a => Int -> Tensor a -> Tensor Int
argmaxAxis = undefined

-- ============================================================
-- Linear Algebra
-- ============================================================

-- | Dot product of two vectors.
--
-- >>> dot [1, 2, 3] [4, 5, 6]
-- 32
foreign import ccall "bhc_tensor_dot"
    dot :: Num a => Tensor a -> Tensor a -> a

-- | Matrix multiplication.
--
-- ==== __Complexity__
--
-- O(n * m * k) for (n x m) @ (m x k) matrices.
foreign import ccall "bhc_tensor_matmul"
    matmul :: Num a => Tensor a -> Tensor a -> Tensor a

-- | Infix matrix multiplication.
(@) :: Num a => Tensor a -> Tensor a -> Tensor a
(@) = matmul
infixl 7 @

-- | Outer product of two vectors.
outer :: Num a => Tensor a -> Tensor a -> Tensor a
outer = undefined

-- | Inner product (generalized dot).
inner :: Num a => Tensor a -> Tensor a -> Tensor a
inner = undefined

-- | Vector/matrix norm.
--
-- Default is L2 (Frobenius) norm.
norm :: Floating a => Tensor a -> a
norm t = sqrt (tSum (tMap (^ 2) t))

-- | Normalize to unit norm.
normalize :: Floating a => Tensor a -> Tensor a
normalize t = tMap (/ norm t) t

-- | Matrix trace (sum of diagonal).
trace :: Num a => Tensor a -> a
trace = undefined

-- | Matrix determinant.
det :: Floating a => Tensor a -> a
det = undefined

-- | Matrix inverse.
inv :: Floating a => Tensor a -> Tensor a
inv = undefined

-- | Solve linear system Ax = b.
solve :: Floating a => Tensor a -> Tensor a -> Tensor a
solve = undefined

-- | Least squares solution.
lstsq :: Floating a => Tensor a -> Tensor a -> Tensor a
lstsq = undefined

-- | Eigenvalue decomposition.
eig :: Floating a => Tensor a -> (Tensor a, Tensor a)
eig = undefined

-- | Singular value decomposition.
svd :: Floating a => Tensor a -> (Tensor a, Tensor a, Tensor a)
svd = undefined

-- | QR decomposition.
qr :: Floating a => Tensor a -> (Tensor a, Tensor a)
qr = undefined

-- | Cholesky decomposition.
cholesky :: Floating a => Tensor a -> Tensor a
cholesky = undefined

-- ============================================================
-- Comparison
-- ============================================================

-- | Element-wise equality.
tEq :: Eq a => Tensor a -> Tensor a -> Tensor Bool
tEq = tZipWith (==)

-- | Element-wise inequality.
tNe :: Eq a => Tensor a -> Tensor a -> Tensor Bool
tNe = tZipWith (/=)

-- | Element-wise less than.
tLt :: Ord a => Tensor a -> Tensor a -> Tensor Bool
tLt = tZipWith (<)

-- | Element-wise less than or equal.
tLe :: Ord a => Tensor a -> Tensor a -> Tensor Bool
tLe = tZipWith (<=)

-- | Element-wise greater than.
tGt :: Ord a => Tensor a -> Tensor a -> Tensor Bool
tGt = tZipWith (>)

-- | Element-wise greater than or equal.
tGe :: Ord a => Tensor a -> Tensor a -> Tensor Bool
tGe = tZipWith (>=)

-- | Check for NaN.
tIsNan :: RealFloat a => Tensor a -> Tensor Bool
tIsNan = tMap isNaN

-- | Check for infinity.
tIsInf :: RealFloat a => Tensor a -> Tensor Bool
tIsInf = tMap isInfinite

-- | Check for finite values.
tIsFinite :: RealFloat a => Tensor a -> Tensor Bool
tIsFinite = tMap (\x -> not (isNaN x || isInfinite x))

-- ============================================================
-- Type Casting
-- ============================================================

-- | Cast tensor to different element type.
cast :: (a -> b) -> Tensor a -> Tensor b
cast = tMap

-- | Cast tensor to specific dtype.
asType :: DType -> Tensor a -> Tensor b
asType = undefined

-- ============================================================
-- Utilities
-- ============================================================

-- | Create a copy of tensor with new memory.
clone :: Tensor a -> IO (Tensor a)
clone = undefined

-- | Ensure tensor is contiguous in memory.
contiguous :: Tensor a -> Tensor a
contiguous = undefined

-- | Convert tensor to flat list.
toList :: Tensor a -> [a]
toList = undefined

-- | Convert tensor to nested lists.
toLists :: Tensor a -> [[a]]
toLists = undefined

-- | Force materialization (prevent fusion).
--
-- Use when you need to reuse intermediate results.
materialize :: Tensor a -> IO (Tensor a)
materialize = undefined

-- | Execute tensor operations within arena scope.
--
-- Temporary allocations are freed when scope exits.
withTensorArena :: (IO a) -> IO a
withTensorArena action = undefined

-- ============================================================
-- Internal Helpers
-- ============================================================

computeStrides :: Shape -> [Int]
computeStrides sh = scanr (*) 1 (tail sh)

insertAt :: Int -> a -> [a] -> [a]
insertAt 0 x xs = x : xs
insertAt n x (y:ys) = y : insertAt (n-1) x ys
insertAt _ x [] = [x]

permuteList :: [Int] -> [a] -> [a]
permuteList perm xs = [xs !! i | i <- perm]
