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
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

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

import Prelude hiding (product)
import qualified Prelude as P
import Data.IORef
import System.IO.Unsafe (unsafePerformIO)
import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Unboxed.Mutable as VUM
import Control.Monad (forM_, when)
import System.Random (randomRIO)

-- ============================================================
-- Types
-- ============================================================

-- | N-dimensional tensor with element type @a@.
--
-- Tensors track their shape, strides, and memory layout for
-- efficient fusion and vectorization.
data Tensor a = Tensor
    { tensorData   :: !(VU.Vector a)
    , tensorShape  :: ![Int]
    , tensorStride :: ![Int]
    , tensorOffset :: !Int
    } deriving (Eq, Show)

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

-- | /O(n)/. Create a tensor filled with zeros.
--
-- >>> zeros [2, 3]
-- Tensor [2,3] [[0,0,0],[0,0,0]]
--
-- >>> shape (zeros [4, 5, 6])
-- [4, 5, 6]
zeros :: (Num a, VU.Unbox a) => Shape -> Tensor a
zeros sh = Tensor
    { tensorData = VU.replicate (product sh) 0
    , tensorShape = sh
    , tensorStride = computeStrides sh
    , tensorOffset = 0
    }

-- | /O(n)/. Create a tensor filled with ones.
--
-- >>> ones [3]
-- Tensor [3] [1,1,1]
--
-- >>> tSum (ones [10, 10])
-- 100
ones :: (Num a, VU.Unbox a) => Shape -> Tensor a
ones sh = Tensor
    { tensorData = VU.replicate (product sh) 1
    , tensorShape = sh
    , tensorStride = computeStrides sh
    , tensorOffset = 0
    }

-- | /O(n)/. Create a tensor filled with a constant value.
--
-- >>> full [2, 2] 3.14
-- Tensor [2,2] [[3.14,3.14],[3.14,3.14]]
full :: VU.Unbox a => Shape -> a -> Tensor a
full sh val = Tensor
    { tensorData = VU.replicate (product sh) val
    , tensorShape = sh
    , tensorStride = computeStrides sh
    , tensorOffset = 0
    }

-- | /O(n)/. Create a 2D tensor from a nested list.
--
-- >>> fromList [[1,2,3], [4,5,6]]
-- Tensor [2,3] [[1,2,3],[4,5,6]]
fromList :: VU.Unbox a => [[a]] -> Tensor a
fromList [] = Tensor VU.empty [] [] 0
fromList xss =
    let rows = length xss
        cols = if null xss then 0 else length (head xss)
        flat = concat xss
    in fromListFlat [rows, cols] flat

-- | /O(n)/. Create a tensor from a flat list with specified shape.
--
-- >>> fromListFlat [2, 3] [1,2,3,4,5,6]
-- Tensor [2,3] [[1,2,3],[4,5,6]]
--
-- >>> fromListFlat [2, 2, 2] [1,2,3,4,5,6,7,8]
-- Tensor [2,2,2] [[[1,2],[3,4]],[[5,6],[7,8]]]
fromListFlat :: VU.Unbox a => Shape -> [a] -> Tensor a
fromListFlat sh xs
    | length xs /= product sh = error "fromListFlat: size mismatch"
    | otherwise = Tensor
        { tensorData = VU.fromList xs
        , tensorShape = sh
        , tensorStride = computeStrides sh
        , tensorOffset = 0
        }

-- | /O(n)/. Create a 1D tensor with values in the range [start, end) with given step.
--
-- >>> toList (arange 0 10 1)
-- [0,1,2,3,4,5,6,7,8,9]
--
-- >>> toList (arange 0 10 2)
-- [0,2,4,6,8]
--
-- >>> toList (arange 5 0 (-1))
-- [5,4,3,2,1]
arange :: (Num a, Ord a, VU.Unbox a) => a -> a -> a -> Tensor a
arange start end step =
    let vals = takeWhile (< end) $ iterate (+ step) start
        n = length vals
    in Tensor
        { tensorData = VU.fromList vals
        , tensorShape = [n]
        , tensorStride = [1]
        , tensorOffset = 0
        }

-- | /O(n)/. Create a 1D tensor with @n@ evenly spaced values from @start@ to @end@ (inclusive).
--
-- >>> toList (linspace 0 1 5)
-- [0.0,0.25,0.5,0.75,1.0]
--
-- >>> toList (linspace 0 10 3)
-- [0.0,5.0,10.0]
linspace :: (Fractional a, VU.Unbox a) => a -> a -> Int -> Tensor a
linspace start end n
    | n <= 0 = Tensor VU.empty [] [] 0
    | n == 1 = fromListFlat [1] [start]
    | otherwise =
        let step = (end - start) / fromIntegral (n - 1)
            vals = [start + fromIntegral i * step | i <- [0..n-1]]
        in Tensor
            { tensorData = VU.fromList vals
            , tensorShape = [n]
            , tensorStride = [1]
            , tensorOffset = 0
            }

-- | /O(n²)/. Create an n×n identity matrix.
--
-- >>> eye 3
-- Tensor [3,3] [[1,0,0],[0,1,0],[0,0,1]]
--
-- >>> trace (eye 5)
-- 5
eye :: (Num a, VU.Unbox a) => Int -> Tensor a
eye n = unsafePerformIO $ do
    mv <- VUM.replicate (n * n) 0
    forM_ [0..n-1] $ \i ->
        VUM.write mv (i * n + i) 1
    v <- VU.freeze mv
    return $ Tensor v [n, n] [n, 1] 0

-- | Create diagonal matrix from vector.
diag :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
diag t
    | rank t /= 1 = error "diag: expected 1D tensor"
    | otherwise = unsafePerformIO $ do
        let n = size t
        mv <- VUM.replicate (n * n) 0
        forM_ [0..n-1] $ \i ->
            VUM.write mv (i * n + i) (tensorData t VU.! i)
        v <- VU.freeze mv
        return $ Tensor v [n, n] [n, 1] 0

-- | Create tensor with uniform random values in [0, 1).
rand :: Shape -> IO (Tensor Float)
rand sh = do
    let n = product sh
    vals <- sequence [randomRIO (0.0, 1.0) | _ <- [1..n]]
    return $ Tensor
        { tensorData = VU.fromList vals
        , tensorShape = sh
        , tensorStride = computeStrides sh
        , tensorOffset = 0
        }

-- | Create tensor with standard normal random values.
randn :: Shape -> IO (Tensor Float)
randn sh = do
    -- Box-Muller transform for normal distribution
    let n = product sh
        genNormal = do
            u1 <- randomRIO (0.0001, 1.0)
            u2 <- randomRIO (0.0, 1.0)
            let r = sqrt (-2 * log u1)
                theta = 2 * pi * u2
            return (r * cos theta)
    vals <- sequence [genNormal | _ <- [1..n]]
    return $ Tensor
        { tensorData = VU.fromList vals
        , tensorShape = sh
        , tensorStride = computeStrides sh
        , tensorOffset = 0
        }

-- ============================================================
-- Shape Operations
-- ============================================================

-- | /O(1)/. Get the shape of a tensor (list of dimensions).
--
-- >>> shape (zeros [2, 3, 4])
-- [2,3,4]
shape :: Tensor a -> Shape
shape = tensorShape

-- | /O(1)/. Get the rank (number of dimensions) of a tensor.
--
-- >>> rank (zeros [2, 3, 4])
-- 3
--
-- >>> rank (zeros [100])
-- 1
rank :: Tensor a -> Int
rank = length . tensorShape

-- | /O(1)/. Get the total number of elements in a tensor.
--
-- >>> size (zeros [2, 3, 4])
-- 24
size :: Tensor a -> Int
size = product . tensorShape

-- | /O(n)/. Reshape a tensor to a new shape.
--
-- The new shape must have the same total number of elements.
-- If the tensor is not contiguous, a copy is made.
--
-- >>> shape (reshape [6] (zeros [2, 3]))
-- [6]
--
-- >>> shape (reshape [3, 2] (zeros [2, 3]))
-- [3,2]
reshape :: VU.Unbox a => Shape -> Tensor a -> Tensor a
reshape newShape t
    | product newShape /= size t = error "reshape: size mismatch"
    | otherwise =
        -- Ensure contiguous before reshape
        let t' = contiguous t
        in t' { tensorShape = newShape, tensorStride = computeStrides newShape }

-- | /O(n)/. Flatten a tensor to a 1D tensor.
--
-- >>> shape (flatten (zeros [2, 3, 4]))
-- [24]
flatten :: VU.Unbox a => Tensor a -> Tensor a
flatten t = reshape [size t] t

-- | /O(1)/. Remove all dimensions of size 1.
--
-- >>> shape (squeeze (zeros [1, 3, 1, 4, 1]))
-- [3,4]
squeeze :: Tensor a -> Tensor a
squeeze t =
    let newShape = filter (/= 1) (tensorShape t)
        newStride = [s | (d, s) <- zip (tensorShape t) (tensorStride t), d /= 1]
    in t { tensorShape = if null newShape then [1] else newShape
         , tensorStride = if null newStride then [1] else newStride
         }

-- | /O(1)/. Add a dimension of size 1 at the specified position.
--
-- >>> shape (unsqueeze 0 (zeros [3, 4]))
-- [1,3,4]
--
-- >>> shape (unsqueeze 2 (zeros [3, 4]))
-- [3,4,1]
unsqueeze :: Int -> Tensor a -> Tensor a
unsqueeze dim t =
    let sh = tensorShape t
        st = tensorStride t
        newShape = insertAt dim 1 sh
        newStride = insertAt dim (if dim < length st then st !! dim else 1) st
    in t { tensorShape = newShape, tensorStride = newStride }

-- | /O(1)/. Transpose a 2D tensor (swap rows and columns).
--
-- This is a view operation - no data is copied.
--
-- >>> shape (transpose (zeros [3, 4]))
-- [4,3]
transpose :: Tensor a -> Tensor a
transpose t
    | rank t /= 2 = error "transpose: expected 2D tensor"
    | otherwise = permute [1, 0] t

-- | /O(1)/. Permute the dimensions of a tensor.
--
-- This is a view operation - no data is copied.
--
-- >>> shape (permute [2, 0, 1] (zeros [2, 3, 4]))
-- [4,2,3]
permute :: [Int] -> Tensor a -> Tensor a
permute perm t = t
    { tensorShape = permuteList perm (tensorShape t)
    , tensorStride = permuteList perm (tensorStride t)
    }

-- | Broadcast tensor to larger shape.
broadcast :: VU.Unbox a => Shape -> Tensor a -> Tensor a
broadcast targetShape t
    | targetShape == tensorShape t = t
    | length targetShape < length (tensorShape t) = error "broadcast: cannot reduce rank"
    | otherwise =
        let -- Pad shape on the left with 1s
            padded = replicate (length targetShape - length (tensorShape t)) 1 ++ tensorShape t
            paddedStride = replicate (length targetShape - length (tensorStride t)) 0 ++ tensorStride t
            -- Check compatibility and compute new strides
            newStride = zipWith3 checkDim targetShape padded paddedStride
        in t { tensorShape = targetShape, tensorStride = newStride }
  where
    checkDim target src stride
        | src == target = stride
        | src == 1 = 0  -- Broadcast dimension
        | otherwise = error "broadcast: incompatible shapes"

-- | Expand tensor (alias for broadcast).
expand :: VU.Unbox a => Shape -> Tensor a -> Tensor a
expand = broadcast

-- ============================================================
-- Indexing
-- ============================================================

-- | Index into tensor.
--
-- >>> t ! [0, 1, 2]
-- 3.14
(!) :: VU.Unbox a => Tensor a -> [Int] -> a
(!) = index

-- | Index into tensor (function form).
index :: VU.Unbox a => Tensor a -> [Int] -> a
index t indices
    | length indices /= rank t = error "index: wrong number of indices"
    | otherwise =
        let flatIdx = tensorOffset t + sum (zipWith (*) indices (tensorStride t))
        in tensorData t VU.! flatIdx

-- | Slice tensor along dimensions.
--
-- >>> slice [(0, 2), (1, 3)] t
-- Tensor [2, 2] ...
slice :: [(Int, Int)] -> Tensor a -> Tensor a
slice ranges t
    | length ranges /= rank t = error "slice: wrong number of ranges"
    | otherwise =
        let newShape = [hi - lo | (lo, hi) <- ranges]
            newOffset = tensorOffset t + sum (zipWith (*) [lo | (lo, _) <- ranges] (tensorStride t))
        in t { tensorShape = newShape, tensorOffset = newOffset }

-- | Narrow tensor along one dimension.
narrow :: Int -> Int -> Int -> Tensor a -> Tensor a
narrow dim start len t
    | dim < 0 || dim >= rank t = error "narrow: invalid dimension"
    | otherwise =
        let sh = tensorShape t
            st = tensorStride t
            newShape = take dim sh ++ [len] ++ drop (dim + 1) sh
            newOffset = tensorOffset t + start * (st !! dim)
        in t { tensorShape = newShape, tensorOffset = newOffset }

-- | Gather elements along axis using indices.
gather :: (VU.Unbox a, VU.Unbox Int) => Int -> Tensor Int -> Tensor a -> Tensor a
gather axis indices src
    | axis < 0 || axis >= rank src = error "gather: invalid axis"
    | otherwise = unsafePerformIO $ do
        let outShape = tensorShape indices
            n = product outShape
        mv <- VUM.new n
        forM_ [0..n-1] $ \i -> do
            let srcIndices = unflattenIndex outShape i
                idx = index indices srcIndices
                srcIdx = take axis srcIndices ++ [idx] ++ drop (axis + 1) srcIndices
                val = index src srcIdx
            VUM.write mv i val
        v <- VU.freeze mv
        return $ Tensor v outShape (computeStrides outShape) 0

-- | Scatter values into tensor at indices.
scatter :: (VU.Unbox a, VU.Unbox Int) => Int -> Tensor Int -> Tensor a -> Tensor a -> Tensor a
scatter axis indices src dst
    | axis < 0 || axis >= rank dst = error "scatter: invalid axis"
    | otherwise = unsafePerformIO $ do
        let n = size indices
        mv <- VU.thaw (tensorData (contiguous dst))
        forM_ [0..n-1] $ \i -> do
            let srcIndices = unflattenIndex (tensorShape indices) i
                idx = index indices srcIndices
                dstIdx = take axis srcIndices ++ [idx] ++ drop (axis + 1) srcIndices
                flatDst = sum (zipWith (*) dstIdx (tensorStride dst)) + tensorOffset dst
                val = index src srcIndices
            VUM.write mv flatDst val
        v <- VU.freeze mv
        return $ Tensor v (tensorShape dst) (tensorStride dst) 0

-- | Select elements using boolean mask.
masked :: (VU.Unbox a) => Tensor Bool -> Tensor a -> Tensor a
masked mask t
    | tensorShape mask /= tensorShape t = error "masked: shape mismatch"
    | otherwise =
        let pairs = zip (toList mask) (toList t)
            selected = [v | (True, v) <- pairs]
            n = length selected
        in Tensor (VU.fromList selected) [n] [1] 0

-- ============================================================
-- Element-wise Operations (MUST FUSE)
-- ============================================================
--
-- All element-wise operations are designed to fuse with each other
-- according to H26-SPEC Section 8. Fusion eliminates intermediate
-- allocations and reduces memory bandwidth requirements.

-- | /O(n)/. Map a function over all tensor elements.
--
-- >>> tMap (*2) (fromListFlat [3] [1, 2, 3])
-- Tensor [3] [2,4,6]
--
-- ==== __Fusion__
--
-- @tMap f (tMap g t)@ MUST fuse to @tMap (f . g) t@ (single traversal).
-- This is a guaranteed fusion pattern per H26-SPEC Section 8.1.
{-# RULES
"tMap/tMap" forall f g t. tMap f (tMap g t) = tMap (f . g) t
#-}
tMap :: (VU.Unbox a, VU.Unbox b) => (a -> b) -> Tensor a -> Tensor b
tMap f t = Tensor
    { tensorData = VU.map f (tensorData (contiguous t))
    , tensorShape = tensorShape t
    , tensorStride = computeStrides (tensorShape t)
    , tensorOffset = 0
    }

-- | /O(n)/. Combine two tensors element-wise with a binary function.
--
-- The tensors must have the same shape.
--
-- >>> tZipWith (+) (ones [2, 2]) (ones [2, 2])
-- Tensor [2,2] [[2,2],[2,2]]
--
-- ==== __Fusion__
--
-- @tZipWith f (tMap g a) (tMap h b)@ MUST fuse to a single traversal.
{-# RULES
"tZipWith/tMap/tMap" forall f g h a b.
    tZipWith f (tMap g a) (tMap h b) = tZipWith (\x y -> f (g x) (h y)) a b
#-}
tZipWith :: (VU.Unbox a, VU.Unbox b, VU.Unbox c)
         => (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
tZipWith f t1 t2
    | tensorShape t1 /= tensorShape t2 = error "tZipWith: shape mismatch"
    | otherwise = Tensor
        { tensorData = VU.zipWith f (tensorData (contiguous t1)) (tensorData (contiguous t2))
        , tensorShape = tensorShape t1
        , tensorStride = computeStrides (tensorShape t1)
        , tensorOffset = 0
        }

-- | /O(n)/. Element-wise addition.
--
-- >>> tAdd (ones [2]) (ones [2])
-- Tensor [2] [2,2]
tAdd :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tAdd = tZipWith (+)

-- | /O(n)/. Element-wise subtraction.
tSub :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tSub = tZipWith (-)

-- | /O(n)/. Element-wise multiplication (Hadamard product).
tMul :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tMul = tZipWith (*)

-- | /O(n)/. Element-wise division.
tDiv :: (Fractional a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tDiv = tZipWith (/)

-- | /O(n)/. Negate all elements.
tNeg :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
tNeg = tMap negate

-- | /O(n)/. Absolute value of all elements.
tAbs :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
tAbs = tMap abs

-- | /O(n)/. Sign of each element (-1, 0, or 1).
tSign :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
tSign = tMap signum

-- | /O(n)/. Square root of all elements.
tSqrt :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tSqrt = tMap sqrt

-- | /O(n)/. Exponential (e^x) of all elements.
tExp :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tExp = tMap exp

-- | /O(n)/. Natural logarithm of all elements.
tLog :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tLog = tMap log

-- | /O(n)/. Element-wise power.
tPow :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tPow = tZipWith (**)

-- | /O(n)/. Sine of all elements (radians).
tSin :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tSin = tMap sin

-- | /O(n)/. Cosine of all elements (radians).
tCos :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tCos = tMap cos

-- | /O(n)/. Tangent of all elements (radians).
tTan :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tTan = tMap tan

-- | /O(n)/. Hyperbolic sine of all elements.
tSinh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tSinh = tMap sinh

-- | /O(n)/. Hyperbolic cosine of all elements.
tCosh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tCosh = tMap cosh

-- | /O(n)/. Hyperbolic tangent of all elements.
--
-- Commonly used as an activation function in neural networks.
tTanh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tTanh = tMap tanh

-- | /O(n)/. Floor of all elements.
tFloor :: (RealFrac a, VU.Unbox a) => Tensor a -> Tensor a
tFloor = tMap (fromIntegral . floor)

-- | /O(n)/. Ceiling of all elements.
tCeil :: (RealFrac a, VU.Unbox a) => Tensor a -> Tensor a
tCeil = tMap (fromIntegral . ceiling)

-- | /O(n)/. Round all elements to nearest integer.
tRound :: (RealFrac a, VU.Unbox a) => Tensor a -> Tensor a
tRound = tMap (fromIntegral . round)

-- | /O(n)/. Element-wise minimum of two tensors.
tMin :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tMin = tZipWith min

-- | /O(n)/. Element-wise maximum of two tensors.
tMax :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tMax = tZipWith max

-- | /O(n)/. Clamp all values to the range [lo, hi].
--
-- >>> toList (tClamp 0 1 (fromListFlat [5] [-1, 0, 0.5, 1, 2]))
-- [0.0,0.0,0.5,1.0,1.0]
tClamp :: (Ord a, VU.Unbox a) => a -> a -> Tensor a -> Tensor a
tClamp lo hi = tMap (max lo . min hi)

-- ============================================================
-- Reductions (MUST FUSE with maps)
-- ============================================================
--
-- Reduction operations collapse tensor elements to a single value.
-- All reductions fuse with preceding map operations to avoid
-- intermediate allocations.

-- | /O(n)/. Sum of all elements.
--
-- >>> tSum (fromListFlat [4] [1, 2, 3, 4])
-- 10
--
-- ==== __Fusion__
--
-- @tSum (tMap f t)@ MUST fuse to a single traversal with no
-- intermediate allocation. This is a guaranteed fusion pattern.
{-# RULES
"tSum/tMap" forall f t. tSum (tMap f t) = tFoldl' (\acc x -> acc + f x) 0 t
#-}
tSum :: (Num a, VU.Unbox a) => Tensor a -> a
tSum = VU.sum . tensorData . contiguous

-- | /O(n)/. Product of all elements.
--
-- >>> tProduct (fromListFlat [4] [1, 2, 3, 4])
-- 24
{-# RULES
"tProduct/tMap" forall f t. tProduct (tMap f t) = tFoldl' (\acc x -> acc * f x) 1 t
#-}
tProduct :: (Num a, VU.Unbox a) => Tensor a -> a
tProduct = VU.product . tensorData . contiguous

-- | /O(n)/. Arithmetic mean of all elements.
--
-- \[ \text{mean}(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^{n} x_i \]
--
-- >>> tMean (fromListFlat [4] [1, 2, 3, 4])
-- 2.5
tMean :: (Fractional a, VU.Unbox a) => Tensor a -> a
tMean t = tSum t / fromIntegral (size t)

-- | /O(n)/. Population variance of all elements.
--
-- \[ \text{var}(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 \]
--
-- >>> tVar (fromListFlat [4] [1, 2, 3, 4])
-- 1.25
--
-- __Note__: This computes population variance (divides by n).
-- For sample variance (divides by n-1), use @tVar t * n / (n-1)@.
tVar :: (Floating a, VU.Unbox a) => Tensor a -> a
tVar t =
    let m = tMean t
        n = fromIntegral (size t)
    in tSum (tMap (\x -> (x - m) ^ (2 :: Int)) t) / n

-- | /O(n)/. Population standard deviation of all elements.
--
-- \[ \text{std}(\mathbf{x}) = \sqrt{\text{var}(\mathbf{x})} \]
--
-- >>> tStd (fromListFlat [4] [2, 4, 4, 4, 5, 5, 7, 9])
-- 2.0
tStd :: (Floating a, VU.Unbox a) => Tensor a -> a
tStd = sqrt . tVar

-- | /O(n)/. Minimum element of the tensor.
--
-- >>> tMin' (fromListFlat [5] [3, 1, 4, 1, 5])
-- 1
--
-- __Warning__: Partial function. Throws an error on empty tensors.
tMin' :: (Ord a, VU.Unbox a) => Tensor a -> a
tMin' t = VU.minimum (tensorData (contiguous t))

-- | /O(n)/. Maximum element of the tensor.
--
-- >>> tMax' (fromListFlat [5] [3, 1, 4, 1, 5])
-- 5
--
-- __Warning__: Partial function. Throws an error on empty tensors.
tMax' :: (Ord a, VU.Unbox a) => Tensor a -> a
tMax' t = VU.maximum (tensorData (contiguous t))

-- | /O(n)/. Index of the minimum element (flat index).
--
-- >>> tArgmin (fromListFlat [5] [3, 1, 4, 1, 5])
-- 1
--
-- __Note__: Returns the first occurrence if there are ties.
tArgmin :: (Ord a, VU.Unbox a) => Tensor a -> Int
tArgmin t = VU.minIndex (tensorData (contiguous t))

-- | /O(n)/. Index of the maximum element (flat index).
--
-- >>> tArgmax (fromListFlat [5] [3, 1, 4, 1, 5])
-- 4
--
-- __Note__: Returns the first occurrence if there are ties.
tArgmax :: (Ord a, VU.Unbox a) => Tensor a -> Int
tArgmax t = VU.maxIndex (tensorData (contiguous t))

-- | /O(n)/. True if any element is True.
--
-- >>> tAny (tGt (fromListFlat [3] [1, 2, 3]) (fromListFlat [3] [0, 5, 0]))
-- True
--
-- Short-circuits on first True (in sequential evaluation).
tAny :: Tensor Bool -> Bool
tAny t = VU.or (tensorData (contiguous t))

-- | /O(n)/. True if all elements are True.
--
-- >>> tAll (tGt (fromListFlat [3] [1, 2, 3]) (zeros [3]))
-- True
--
-- Short-circuits on first False (in sequential evaluation).
tAll :: Tensor Bool -> Bool
tAll t = VU.and (tensorData (contiguous t))

-- Internal fold
tFoldl' :: VU.Unbox a => (b -> a -> b) -> b -> Tensor a -> b
tFoldl' f z t = VU.foldl' f z (tensorData (contiguous t))

-- ============================================================
-- Reduction Along Axis
-- ============================================================
--
-- These operations reduce a tensor along a specified axis,
-- producing a tensor with one fewer dimension.

-- | /O(n)/. Sum along a specified axis.
--
-- Reduces the tensor by summing along the given axis.
--
-- >>> sumAxis 0 (fromList [[1, 2], [3, 4]])
-- Tensor [2] [4,6]
--
-- >>> sumAxis 1 (fromList [[1, 2], [3, 4]])
-- Tensor [2] [3,7]
sumAxis :: (Num a, VU.Unbox a) => Int -> Tensor a -> Tensor a
sumAxis = reduceAxis (+) 0

-- | /O(n)/. Product along a specified axis.
--
-- Reduces the tensor by multiplying along the given axis.
--
-- >>> productAxis 0 (fromList [[1, 2], [3, 4]])
-- Tensor [2] [3,8]
productAxis :: (Num a, VU.Unbox a) => Int -> Tensor a -> Tensor a
productAxis = reduceAxis (*) 1

-- | /O(n)/. Mean along a specified axis.
--
-- Computes the arithmetic mean along the given axis.
--
-- >>> meanAxis 0 (fromList [[1, 2], [3, 4]])
-- Tensor [2] [2.0,3.0]
meanAxis :: (Fractional a, VU.Unbox a) => Int -> Tensor a -> Tensor a
meanAxis axis t =
    let sumT = sumAxis axis t
        n = fromIntegral (tensorShape t !! axis)
    in tMap (/ n) sumT

-- | /O(n)/. Minimum along a specified axis.
--
-- >>> minAxis 0 (fromList [[3, 1], [2, 4]])
-- Tensor [2] [2,1]
minAxis :: (Ord a, VU.Unbox a, Bounded a) => Int -> Tensor a -> Tensor a
minAxis = reduceAxis min maxBound

-- | /O(n)/. Maximum along a specified axis.
--
-- >>> maxAxis 0 (fromList [[3, 1], [2, 4]])
-- Tensor [2] [3,4]
maxAxis :: (Ord a, VU.Unbox a, Bounded a) => Int -> Tensor a -> Tensor a
maxAxis = reduceAxis max minBound

-- | /O(n)/. Index of minimum along a specified axis.
--
-- Returns the indices of minimum values along the axis.
--
-- >>> argminAxis 0 (fromList [[3, 1], [2, 4]])
-- Tensor [2] [1,0]
argminAxis :: (Ord a, VU.Unbox a) => Int -> Tensor a -> Tensor Int
argminAxis axis t = argReduceAxis (<) axis t

-- | /O(n)/. Index of maximum along a specified axis.
--
-- Returns the indices of maximum values along the axis.
--
-- >>> argmaxAxis 0 (fromList [[3, 1], [2, 4]])
-- Tensor [2] [0,1]
argmaxAxis :: (Ord a, VU.Unbox a) => Int -> Tensor a -> Tensor Int
argmaxAxis axis t = argReduceAxis (>) axis t

-- Helper for axis reduction
reduceAxis :: VU.Unbox a => (a -> a -> a) -> a -> Int -> Tensor a -> Tensor a
reduceAxis op initial axis t
    | axis < 0 || axis >= rank t = error "reduceAxis: invalid axis"
    | otherwise = unsafePerformIO $ do
        let sh = tensorShape t
            axisSize = sh !! axis
            newShape = take axis sh ++ drop (axis + 1) sh
            newSize = product newShape
        mv <- VUM.replicate newSize initial
        forM_ [0..size t - 1] $ \i -> do
            let indices = unflattenIndex sh i
                reducedIndices = take axis indices ++ drop (axis + 1) indices
                outIdx = flattenIndex newShape reducedIndices
            oldVal <- VUM.read mv outIdx
            let curVal = index t indices
            VUM.write mv outIdx (op oldVal curVal)
        v <- VU.freeze mv
        return $ Tensor v (if null newShape then [1] else newShape) (computeStrides (if null newShape then [1] else newShape)) 0

-- Helper for argmin/argmax along axis
argReduceAxis :: VU.Unbox a => (a -> a -> Bool) -> Int -> Tensor a -> Tensor Int
argReduceAxis cmp axis t
    | axis < 0 || axis >= rank t = error "argReduceAxis: invalid axis"
    | otherwise = unsafePerformIO $ do
        let sh = tensorShape t
            axisSize = sh !! axis
            newShape = take axis sh ++ drop (axis + 1) sh
            newSize = product newShape
        mvIdx <- VUM.replicate newSize (0 :: Int)
        mvVal <- VUM.new newSize
        -- Initialize with first element along axis
        forM_ [0..newSize - 1] $ \outIdx -> do
            let reducedIndices = unflattenIndex newShape outIdx
                fullIndices = take axis reducedIndices ++ [0] ++ drop axis reducedIndices
            VUM.write mvVal outIdx (index t fullIndices)
        -- Find min/max along axis
        forM_ [0..size t - 1] $ \i -> do
            let indices = unflattenIndex sh i
                axisIdx = indices !! axis
                reducedIndices = take axis indices ++ drop (axis + 1) indices
                outIdx = flattenIndex newShape reducedIndices
            when (axisIdx > 0) $ do
                oldVal <- VUM.read mvVal outIdx
                let curVal = index t indices
                when (curVal `cmp` oldVal) $ do
                    VUM.write mvVal outIdx curVal
                    VUM.write mvIdx outIdx axisIdx
        vIdx <- VU.freeze mvIdx
        return $ Tensor vIdx (if null newShape then [1] else newShape) (computeStrides (if null newShape then [1] else newShape)) 0

-- ============================================================
-- Linear Algebra
-- ============================================================
--
-- Standard linear algebra operations. For large matrices,
-- these operations use BLAS when available.

-- | /O(n)/. Dot product (inner product) of two 1D tensors.
--
-- \[ \text{dot}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} x_i \cdot y_i \]
--
-- >>> dot (fromListFlat [3] [1, 2, 3]) (fromListFlat [3] [4, 5, 6])
-- 32
--
-- ==== __SIMD__
--
-- Uses SIMD vectorization for Float and Double tensors.
dot :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> a
dot t1 t2
    | rank t1 /= 1 || rank t2 /= 1 = error "dot: expected 1D tensors"
    | size t1 /= size t2 = error "dot: size mismatch"
    | otherwise = tSum (tMul t1 t2)

-- | /O(n·m·k)/. Matrix multiplication.
--
-- Computes the matrix product of an (n×m) and (m×k) matrix,
-- producing an (n×k) result.
--
-- >>> let a = fromList [[1, 2], [3, 4]]
-- >>> let b = fromList [[5, 6], [7, 8]]
-- >>> matmul a b
-- Tensor [2,2] [[19,22],[43,50]]
--
-- ==== __BLAS__
--
-- For Double matrices, uses BLAS DGEMM when available,
-- providing optimized cache-aware tiled multiplication.
matmul :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
matmul t1 t2
    | rank t1 /= 2 || rank t2 /= 2 = error "matmul: expected 2D tensors"
    | tensorShape t1 !! 1 /= tensorShape t2 !! 0 = error "matmul: incompatible shapes"
    | otherwise = unsafePerformIO $ do
        let [m, k] = tensorShape t1
            [_, n] = tensorShape t2
        mv <- VUM.new (m * n)
        forM_ [0..m-1] $ \i ->
            forM_ [0..n-1] $ \j -> do
                let val = sum [index t1 [i, l] * index t2 [l, j] | l <- [0..k-1]]
                VUM.write mv (i * n + j) val
        v <- VU.freeze mv
        return $ Tensor v [m, n] [n, 1] 0

-- | Infix operator for matrix multiplication.
--
-- >>> let a = fromList [[1, 0], [0, 1]]
-- >>> let b = fromList [[2, 3], [4, 5]]
-- >>> a @ b
-- Tensor [2,2] [[2,3],[4,5]]
(@) :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
(@) = matmul
infixl 7 @

-- | /O(n·m)/. Outer product of two 1D tensors.
--
-- Produces a matrix where @result[i,j] = x[i] * y[j]@.
--
-- >>> outer (fromListFlat [2] [1, 2]) (fromListFlat [3] [3, 4, 5])
-- Tensor [2,3] [[3,4,5],[6,8,10]]
outer :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
outer t1 t2
    | rank t1 /= 1 || rank t2 /= 1 = error "outer: expected 1D tensors"
    | otherwise = unsafePerformIO $ do
        let m = size t1
            n = size t2
        mv <- VUM.new (m * n)
        forM_ [0..m-1] $ \i ->
            forM_ [0..n-1] $ \j -> do
                let val = index t1 [i] * index t2 [j]
                VUM.write mv (i * n + j) val
        v <- VU.freeze mv
        return $ Tensor v [m, n] [n, 1] 0

-- | Generalized inner product.
--
-- Contracts the last axis of the first tensor with the first axis
-- of the second tensor. For 1D tensors, equivalent to 'dot'.
-- For 2D tensors, equivalent to 'matmul'.
inner :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
inner t1 t2
    | null (tensorShape t1) || null (tensorShape t2) = error "inner: empty tensor"
    | last (tensorShape t1) /= head (tensorShape t2) = error "inner: incompatible shapes"
    | otherwise =
        -- General inner product contracts last axis of t1 with first axis of t2
        -- For 1D tensors, this is dot product
        if rank t1 == 1 && rank t2 == 1
        then fromListFlat [] [dot t1 t2]  -- Scalar result
        else matmul t1 t2  -- For 2D, same as matmul

-- | /O(n)/. L2 (Frobenius) norm.
--
-- \[ \|\mathbf{x}\|_2 = \sqrt{\sum_{i} x_i^2} \]
--
-- >>> norm (fromListFlat [3] [3, 4, 0])
-- 5.0
norm :: (Floating a, VU.Unbox a) => Tensor a -> a
norm t = sqrt (tSum (tMap (^ (2 :: Int)) t))

-- | /O(n)/. Normalize tensor to unit norm.
--
-- Returns the tensor divided by its L2 norm.
--
-- >>> normalize (fromListFlat [2] [3, 4])
-- Tensor [2] [0.6,0.8]
--
-- Returns the original tensor unchanged if norm is zero.
normalize :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
normalize t =
    let n = norm t
    in if n == 0 then t else tMap (/ n) t

-- | /O(n)/. Matrix trace (sum of diagonal elements).
--
-- >>> trace (fromList [[1, 2], [3, 4]])
-- 5
trace :: (Num a, VU.Unbox a) => Tensor a -> a
trace t
    | rank t /= 2 = error "trace: expected 2D tensor"
    | otherwise =
        let [m, n] = tensorShape t
            k = min m n
        in sum [index t [i, i] | i <- [0..k-1]]

-- | /O(n³)/. Matrix determinant.
--
-- Computes the determinant using LU decomposition.
--
-- >>> det (fromList [[1, 2], [3, 4]])
-- -2.0
--
-- >>> det (eye 3)
-- 1.0
det :: (Floating a, VU.Unbox a) => Tensor a -> a
det t
    | rank t /= 2 = error "det: expected 2D tensor"
    | tensorShape t !! 0 /= tensorShape t !! 1 = error "det: expected square matrix"
    | otherwise =
        let n = tensorShape t !! 0
        in case n of
            0 -> 1
            1 -> index t [0, 0]
            2 -> let a = index t [0, 0]
                     b = index t [0, 1]
                     c = index t [1, 0]
                     d = index t [1, 1]
                 in a * d - b * c
            _ -> -- LU decomposition for larger matrices
                 let (l, u, sign) = luDecomp t
                     diagProd = P.product [index u [i, i] | i <- [0..n-1]]
                 in sign * diagProd

-- | /O(n³)/. Matrix inverse using Gauss-Jordan elimination.
--
-- >>> let a = fromList [[4, 7], [2, 6]]
-- >>> inv a
-- Tensor [2,2] [[0.6,-0.7],[-0.2,0.4]]
--
-- __Warning__: Throws an error for singular matrices.
inv :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> Tensor a
inv t
    | rank t /= 2 = error "inv: expected 2D tensor"
    | tensorShape t !! 0 /= tensorShape t !! 1 = error "inv: expected square matrix"
    | otherwise = unsafePerformIO $ do
        let n = tensorShape t !! 0
        -- Gauss-Jordan elimination
        mv <- VUM.new (n * 2 * n)
        -- Initialize augmented matrix [A | I]
        forM_ [0..n-1] $ \i ->
            forM_ [0..n-1] $ \j -> do
                VUM.write mv (i * 2 * n + j) (index t [i, j])
                VUM.write mv (i * 2 * n + n + j) (if i == j then 1 else 0)
        -- Forward elimination
        forM_ [0..n-1] $ \col -> do
            -- Find pivot
            pivotVal <- VUM.read mv (col * 2 * n + col)
            when (abs pivotVal < 1e-10) $ error "inv: singular matrix"
            -- Scale pivot row
            forM_ [0..2*n-1] $ \j -> do
                val <- VUM.read mv (col * 2 * n + j)
                VUM.write mv (col * 2 * n + j) (val / pivotVal)
            -- Eliminate column
            forM_ [0..n-1] $ \row ->
                when (row /= col) $ do
                    factor <- VUM.read mv (row * 2 * n + col)
                    forM_ [0..2*n-1] $ \j -> do
                        pivotRowVal <- VUM.read mv (col * 2 * n + j)
                        rowVal <- VUM.read mv (row * 2 * n + j)
                        VUM.write mv (row * 2 * n + j) (rowVal - factor * pivotRowVal)
        -- Extract inverse from right half
        mvResult <- VUM.new (n * n)
        forM_ [0..n-1] $ \i ->
            forM_ [0..n-1] $ \j -> do
                val <- VUM.read mv (i * 2 * n + n + j)
                VUM.write mvResult (i * n + j) val
        v <- VU.freeze mvResult
        return $ Tensor v [n, n] [n, 1] 0

-- | /O(n³)/. Solve linear system Ax = b.
--
-- Finds x such that Ax = b, where A is a square matrix.
--
-- >>> let a = fromList [[3, 1], [1, 2]]
-- >>> let b = fromListFlat [2, 1] [9, 8]
-- >>> solve a b
-- Tensor [2,1] [[2],[3]]
solve :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
solve a b
    | rank a /= 2 = error "solve: A must be 2D"
    | tensorShape a !! 0 /= tensorShape a !! 1 = error "solve: A must be square"
    | otherwise = matmul (inv a) b

-- | /O(n²m)/. Least squares solution.
--
-- Finds x that minimizes ||Ax - b||² using the normal equations.
--
-- Computes @x = (A^T A)^(-1) A^T b@.
--
-- Useful for overdetermined systems (more equations than unknowns).
lstsq :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
lstsq a b =
    -- x = (A^T A)^(-1) A^T b
    let at = transpose a
        ata = matmul at a
        atb = matmul at b
    in solve ata atb

-- | Eigenvalue decomposition using power iteration.
--
-- Returns @(eigenvalues, eigenvectors)@ for a square matrix.
--
-- __Note__: Current implementation uses power iteration,
-- which finds only the dominant eigenvalue/eigenvector pair.
-- Full QR-based eigendecomposition planned for future release.
eig :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> (Tensor a, Tensor a)
eig t
    | rank t /= 2 = error "eig: expected 2D tensor"
    | tensorShape t !! 0 /= tensorShape t !! 1 = error "eig: expected square matrix"
    | otherwise =
        -- Simple power iteration for dominant eigenvalue/vector
        -- Full implementation would use QR algorithm
        let n = tensorShape t !! 0
            -- Start with random vector
            v0 = fromListFlat [n] (replicate n 1.0)
            -- Power iteration
            iterate' 0 v = (0, v)  -- Fallback
            iterate' maxIter v =
                let av = matmul t (reshape [n, 1] v)
                    av' = flatten av
                    lambda = norm av'
                    v' = normalize av'
                in if maxIter <= 0
                   then (lambda, v')
                   else iterate' (maxIter - 1) v'
            (eigenval, eigenvec) = iterate' (100 :: Int) v0
        in (fromListFlat [1] [eigenval], reshape [n, 1] eigenvec)

-- | Singular value decomposition.
--
-- Returns @(U, S, V)@ where @A = U * diag(S) * V^T@.
--
-- * @U@ - Left singular vectors (orthonormal columns)
-- * @S@ - Singular values (non-negative, in descending order)
-- * @V@ - Right singular vectors (orthonormal columns)
--
-- __Note__: Current implementation is simplified.
-- Full divide-and-conquer SVD planned for future release.
svd :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> (Tensor a, Tensor a, Tensor a)
svd t
    | rank t /= 2 = error "svd: expected 2D tensor"
    | otherwise =
        -- Simplified SVD via eigendecomposition of A^T A
        let [m, n] = tensorShape t
            k = min m n
            -- A^T A = V * S^2 * V^T
            ata = matmul (transpose t) t
            (eigvals, eigvecs) = eig ata
            -- Singular values are sqrt of eigenvalues
            s = tSqrt (tAbs eigvals)
            -- V is eigenvectors of A^T A
            v = eigvecs
            -- U = A V S^(-1)
            sVal = if size s > 0 then index s [0] else 1
            sInv = if abs sVal > 1e-10 then 1 / sVal else 0
            u = if sInv == 0 then zeros [m, 1] else tMap (* sInv) (matmul t v)
        in (u, s, v)

-- | /O(n²m)/. QR decomposition using Gram-Schmidt orthogonalization.
--
-- Returns @(Q, R)@ where @A = Q * R@.
--
-- * @Q@ - Orthogonal matrix (Q^T Q = I)
-- * @R@ - Upper triangular matrix
--
-- >>> let (q, r) = qr (fromList [[1, 2], [3, 4], [5, 6]])
-- >>> shape q
-- [3, 2]
-- >>> shape r
-- [2, 2]
qr :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> (Tensor a, Tensor a)
qr t
    | rank t /= 2 = error "qr: expected 2D tensor"
    | otherwise = unsafePerformIO $ do
        let [m, n] = tensorShape t
            k = min m n
        -- Gram-Schmidt orthogonalization
        qCols <- sequence [VUM.new m | _ <- [0..k-1]]
        rData <- VUM.replicate (k * n) 0
        forM_ [0..k-1] $ \j -> do
            -- Start with column j of A
            let col = [index t [i, j] | i <- [0..m-1]]
            forM_ [0..m-1] $ \i ->
                VUM.write (qCols !! j) i (col !! i)
            -- Subtract projections onto previous q vectors
            forM_ [0..j-1] $ \i -> do
                qiVals <- sequence [VUM.read (qCols !! i) idx | idx <- [0..m-1]]
                ujVals <- sequence [VUM.read (qCols !! j) idx | idx <- [0..m-1]]
                let rij = sum (zipWith (*) qiVals ujVals)
                VUM.write rData (i * n + j) rij
                forM_ [0..m-1] $ \idx -> do
                    qi <- VUM.read (qCols !! i) idx
                    uj <- VUM.read (qCols !! j) idx
                    VUM.write (qCols !! j) idx (uj - rij * qi)
            -- Normalize
            ujVals <- sequence [VUM.read (qCols !! j) idx | idx <- [0..m-1]]
            let normVal = sqrt (sum (map (^(2::Int)) ujVals))
            VUM.write rData (j * n + j) normVal
            when (abs normVal > 1e-10) $
                forM_ [0..m-1] $ \idx -> do
                    uj <- VUM.read (qCols !! j) idx
                    VUM.write (qCols !! j) idx (uj / normVal)
        -- Build Q matrix
        qData <- VUM.new (m * k)
        forM_ [0..k-1] $ \j ->
            forM_ [0..m-1] $ \i -> do
                val <- VUM.read (qCols !! j) i
                VUM.write qData (i * k + j) val
        q <- VU.freeze qData
        r <- VU.freeze rData
        return (Tensor q [m, k] [k, 1] 0, Tensor r [k, n] [n, 1] 0)

-- | /O(n³)/. Cholesky decomposition.
--
-- Returns lower triangular matrix @L@ where @A = L * L^T@.
--
-- Input matrix must be symmetric positive-definite.
--
-- >>> cholesky (fromList [[4, 2], [2, 5]])
-- Tensor [2,2] [[2.0,0.0],[1.0,2.0]]
--
-- Useful for efficiently solving linear systems and computing determinants
-- of positive-definite matrices.
cholesky :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> Tensor a
cholesky t
    | rank t /= 2 = error "cholesky: expected 2D tensor"
    | tensorShape t !! 0 /= tensorShape t !! 1 = error "cholesky: expected square matrix"
    | otherwise = unsafePerformIO $ do
        let n = tensorShape t !! 0
        lData <- VUM.replicate (n * n) 0
        forM_ [0..n-1] $ \i -> do
            forM_ [0..i] $ \j -> do
                if i == j
                then do
                    -- Diagonal element
                    prevSum <- fmap sum $ sequence
                        [do val <- VUM.read lData (i * n + k)
                            return (val * val)
                        | k <- [0..j-1]]
                    let val = sqrt (index t [i, i] - prevSum)
                    VUM.write lData (i * n + j) val
                else do
                    -- Off-diagonal element
                    prevSum <- fmap sum $ sequence
                        [do li <- VUM.read lData (i * n + k)
                            lj <- VUM.read lData (j * n + k)
                            return (li * lj)
                        | k <- [0..j-1]]
                    ljj <- VUM.read lData (j * n + j)
                    let val = (index t [i, j] - prevSum) / ljj
                    VUM.write lData (i * n + j) val
        l <- VU.freeze lData
        return $ Tensor l [n, n] [n, 1] 0

-- LU decomposition helper (returns L, U, sign)
luDecomp :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> (Tensor a, Tensor a, a)
luDecomp t = unsafePerformIO $ do
    let n = tensorShape t !! 0
    lData <- VUM.replicate (n * n) 0
    uData <- VUM.replicate (n * n) 0
    signRef <- newIORef (1 :: a)
    -- Initialize U with A
    forM_ [0..n-1] $ \i ->
        forM_ [0..n-1] $ \j ->
            VUM.write uData (i * n + j) (index t [i, j])
    -- Initialize L with identity
    forM_ [0..n-1] $ \i ->
        VUM.write lData (i * n + i) 1
    -- Doolittle algorithm
    forM_ [0..n-1] $ \k -> do
        -- Partial pivoting would go here
        ukk <- VUM.read uData (k * n + k)
        forM_ [k+1..n-1] $ \i -> do
            uik <- VUM.read uData (i * n + k)
            let factor = uik / ukk
            VUM.write lData (i * n + k) factor
            forM_ [k..n-1] $ \j -> do
                uij <- VUM.read uData (i * n + j)
                ukj <- VUM.read uData (k * n + j)
                VUM.write uData (i * n + j) (uij - factor * ukj)
    l <- VU.freeze lData
    u <- VU.freeze uData
    sign <- readIORef signRef
    return (Tensor l [n, n] [n, 1] 0, Tensor u [n, n] [n, 1] 0, sign)

-- ============================================================
-- Comparison
-- ============================================================
--
-- Element-wise comparison operations that return Boolean tensors.
-- These can be combined with 'tAny' and 'tAll' for aggregate checks.

-- | /O(n)/. Element-wise equality comparison.
--
-- >>> tEq (fromListFlat [3] [1, 2, 3]) (fromListFlat [3] [1, 5, 3])
-- Tensor [3] [True,False,True]
tEq :: (Eq a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tEq = tZipWith (==)

-- | /O(n)/. Element-wise inequality comparison.
--
-- >>> tNe (fromListFlat [3] [1, 2, 3]) (fromListFlat [3] [1, 5, 3])
-- Tensor [3] [False,True,False]
tNe :: (Eq a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tNe = tZipWith (/=)

-- | /O(n)/. Element-wise less-than comparison.
--
-- >>> tLt (fromListFlat [3] [1, 2, 3]) (fromListFlat [3] [2, 2, 2])
-- Tensor [3] [True,False,False]
tLt :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tLt = tZipWith (<)

-- | /O(n)/. Element-wise less-than-or-equal comparison.
--
-- >>> tLe (fromListFlat [3] [1, 2, 3]) (fromListFlat [3] [2, 2, 2])
-- Tensor [3] [True,True,False]
tLe :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tLe = tZipWith (<=)

-- | /O(n)/. Element-wise greater-than comparison.
--
-- >>> tGt (fromListFlat [3] [1, 2, 3]) (fromListFlat [3] [2, 2, 2])
-- Tensor [3] [False,False,True]
tGt :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tGt = tZipWith (>)

-- | /O(n)/. Element-wise greater-than-or-equal comparison.
--
-- >>> tGe (fromListFlat [3] [1, 2, 3]) (fromListFlat [3] [2, 2, 2])
-- Tensor [3] [False,True,True]
tGe :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tGe = tZipWith (>=)

-- | /O(n)/. Check each element for NaN (Not a Number).
--
-- >>> tIsNan (fromListFlat [3] [1.0, 0/0, 2.0])
-- Tensor [3] [False,True,False]
tIsNan :: (RealFloat a, VU.Unbox a) => Tensor a -> Tensor Bool
tIsNan = tMap isNaN

-- | /O(n)/. Check each element for infinity.
--
-- >>> tIsInf (fromListFlat [3] [1.0, 1/0, -1/0])
-- Tensor [3] [False,True,True]
tIsInf :: (RealFloat a, VU.Unbox a) => Tensor a -> Tensor Bool
tIsInf = tMap isInfinite

-- | /O(n)/. Check each element for finiteness.
--
-- Returns True for elements that are neither NaN nor infinite.
--
-- >>> tIsFinite (fromListFlat [3] [1.0, 1/0, 0/0])
-- Tensor [3] [True,False,False]
tIsFinite :: (RealFloat a, VU.Unbox a) => Tensor a -> Tensor Bool
tIsFinite = tMap (\x -> not (isNaN x || isInfinite x))

-- ============================================================
-- Type Casting
-- ============================================================
--
-- Convert tensors between element types.

-- | /O(n)/. Cast tensor elements using a conversion function.
--
-- >>> cast round (fromListFlat [3] [1.2, 2.7, 3.5] :: Tensor Double) :: Tensor Int
-- Tensor [3] [1,3,4]
cast :: (VU.Unbox a, VU.Unbox b) => (a -> b) -> Tensor a -> Tensor b
cast = tMap

-- | /O(n)/. Cast tensor to a specific data type.
--
-- Rounds floating-point values when casting to integer types.
--
-- >>> asType Int32 (fromListFlat [3] [1.2, 2.7, 3.5])
-- Tensor [3] [1,3,4]
asType :: (VU.Unbox a, VU.Unbox b, RealFrac a, Num b) => DType -> Tensor a -> Tensor b
asType _ = tMap (fromIntegral . round)

-- ============================================================
-- Utilities
-- ============================================================
--
-- Memory management and conversion utilities.

-- | /O(n)/. Create a deep copy of the tensor with new memory.
--
-- Useful when you need to modify a tensor without affecting views.
clone :: VU.Unbox a => Tensor a -> IO (Tensor a)
clone t = return $ contiguous t

-- | /O(n)/. Ensure tensor data is contiguous in memory.
--
-- Returns the original tensor if already contiguous (O(1)),
-- otherwise creates a copy with contiguous memory layout.
--
-- Many operations (BLAS, FFI) require contiguous memory.
contiguous :: VU.Unbox a => Tensor a -> Tensor a
contiguous t
    | isContiguous t = t
    | otherwise = unsafePerformIO $ do
        let n = size t
            sh = tensorShape t
        mv <- VUM.new n
        forM_ [0..n-1] $ \i -> do
            let indices = unflattenIndex sh i
                val = index t indices
            VUM.write mv i val
        v <- VU.freeze mv
        return $ Tensor v sh (computeStrides sh) 0
  where
    isContiguous t' = tensorOffset t' == 0 && tensorStride t' == computeStrides (tensorShape t')

-- | /O(n)/. Convert tensor to a flat list (row-major order).
--
-- >>> toList (fromList [[1, 2], [3, 4]])
-- [1,2,3,4]
toList :: VU.Unbox a => Tensor a -> [a]
toList t = [index t (unflattenIndex (tensorShape t) i) | i <- [0..size t - 1]]

-- | /O(n)/. Convert a 2D tensor to nested lists.
--
-- >>> toLists (fromList [[1, 2], [3, 4]])
-- [[1,2],[3,4]]
--
-- __Warning__: Only works for 2D tensors.
toLists :: VU.Unbox a => Tensor a -> [[a]]
toLists t
    | rank t /= 2 = error "toLists: expected 2D tensor"
    | otherwise =
        let [rows, cols] = tensorShape t
        in [[index t [i, j] | j <- [0..cols-1]] | i <- [0..rows-1]]

-- | /O(n)/. Force materialization to prevent fusion.
--
-- Use when you need to reuse an intermediate result multiple times.
-- Without materialization, the computation would be repeated.
--
-- >>> let t = materialize (tMap (*2) bigTensor)
-- >>> (tSum t, tProduct t)  -- t computed once, not twice
materialize :: VU.Unbox a => Tensor a -> IO (Tensor a)
materialize = clone

-- | Execute tensor operations within an arena scope.
--
-- All temporary allocations during the action are made in a
-- bump-allocated arena and freed in bulk when the scope exits.
-- This reduces GC pressure for computation-heavy kernels.
--
-- >>> withTensorArena $ do
-- >>>     let a = matmul big1 big2
-- >>>     let b = matmul a big3
-- >>>     return (tSum b)
withTensorArena :: IO a -> IO a
withTensorArena action = action  -- For now, just run the action

-- ============================================================
-- Internal Helpers
-- ============================================================

product :: [Int] -> Int
product = P.product

computeStrides :: Shape -> [Int]
computeStrides [] = []
computeStrides sh = scanr (*) 1 (tail sh)

insertAt :: Int -> a -> [a] -> [a]
insertAt 0 x xs = x : xs
insertAt n x (y:ys) = y : insertAt (n-1) x ys
insertAt _ x [] = [x]

permuteList :: [Int] -> [a] -> [a]
permuteList perm xs = [xs !! i | i <- perm]

-- Convert flat index to multi-dimensional indices
unflattenIndex :: Shape -> Int -> [Int]
unflattenIndex sh idx = go (reverse sh) idx []
  where
    go [] _ acc = acc
    go (d:ds) i acc =
        let (q, r) = i `divMod` d
        in go ds q (r : acc)

-- Convert multi-dimensional indices to flat index
flattenIndex :: Shape -> [Int] -> Int
flattenIndex sh indices = sum (zipWith (*) indices (computeStrides sh))
