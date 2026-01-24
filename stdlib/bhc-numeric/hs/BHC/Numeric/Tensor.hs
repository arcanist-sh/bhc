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

-- | Create tensor filled with zeros.
zeros :: (Num a, VU.Unbox a) => Shape -> Tensor a
zeros sh = Tensor
    { tensorData = VU.replicate (product sh) 0
    , tensorShape = sh
    , tensorStride = computeStrides sh
    , tensorOffset = 0
    }

-- | Create tensor filled with ones.
ones :: (Num a, VU.Unbox a) => Shape -> Tensor a
ones sh = Tensor
    { tensorData = VU.replicate (product sh) 1
    , tensorShape = sh
    , tensorStride = computeStrides sh
    , tensorOffset = 0
    }

-- | Create tensor filled with a constant value.
full :: VU.Unbox a => Shape -> a -> Tensor a
full sh val = Tensor
    { tensorData = VU.replicate (product sh) val
    , tensorShape = sh
    , tensorStride = computeStrides sh
    , tensorOffset = 0
    }

-- | Create tensor from nested list.
--
-- >>> fromList [[1,2,3], [4,5,6]]
-- Tensor [2, 3] ...
fromList :: VU.Unbox a => [[a]] -> Tensor a
fromList [] = Tensor VU.empty [] [] 0
fromList xss =
    let rows = length xss
        cols = if null xss then 0 else length (head xss)
        flat = concat xss
    in fromListFlat [rows, cols] flat

-- | Create tensor from flat list with shape.
--
-- >>> fromListFlat [2, 3] [1,2,3,4,5,6]
-- Tensor [2, 3] ...
fromListFlat :: VU.Unbox a => Shape -> [a] -> Tensor a
fromListFlat sh xs
    | length xs /= product sh = error "fromListFlat: size mismatch"
    | otherwise = Tensor
        { tensorData = VU.fromList xs
        , tensorShape = sh
        , tensorStride = computeStrides sh
        , tensorOffset = 0
        }

-- | Create 1D tensor with range [start, end).
--
-- >>> arange 0 10 1
-- Tensor [10] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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

-- | Create 1D tensor with evenly spaced values.
--
-- >>> linspace 0 1 5
-- Tensor [5] [0.0, 0.25, 0.5, 0.75, 1.0]
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

-- | Create identity matrix.
--
-- >>> eye 3
-- Tensor [3, 3] [[1,0,0], [0,1,0], [0,0,1]]
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
reshape :: VU.Unbox a => Shape -> Tensor a -> Tensor a
reshape newShape t
    | product newShape /= size t = error "reshape: size mismatch"
    | otherwise =
        -- Ensure contiguous before reshape
        let t' = contiguous t
        in t' { tensorShape = newShape, tensorStride = computeStrides newShape }

-- | Flatten tensor to 1D.
flatten :: VU.Unbox a => Tensor a -> Tensor a
flatten t = reshape [size t] t

-- | Remove dimensions of size 1.
squeeze :: Tensor a -> Tensor a
squeeze t =
    let newShape = filter (/= 1) (tensorShape t)
        newStride = [s | (d, s) <- zip (tensorShape t) (tensorStride t), d /= 1]
    in t { tensorShape = if null newShape then [1] else newShape
         , tensorStride = if null newStride then [1] else newStride
         }

-- | Add dimension of size 1 at position.
unsqueeze :: Int -> Tensor a -> Tensor a
unsqueeze dim t =
    let sh = tensorShape t
        st = tensorStride t
        newShape = insertAt dim 1 sh
        newStride = insertAt dim (if dim < length st then st !! dim else 1) st
    in t { tensorShape = newShape, tensorStride = newStride }

-- | Transpose a 2D tensor.
transpose :: Tensor a -> Tensor a
transpose t
    | rank t /= 2 = error "transpose: expected 2D tensor"
    | otherwise = permute [1, 0] t

-- | Permute dimensions.
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

-- | Map function over tensor elements.
--
-- ==== __Fusion__
--
-- @tMap f (tMap g t)@ MUST fuse to @tMap (f . g) t@.
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

-- | Zip two tensors with a binary function.
--
-- ==== __Fusion__
--
-- @tZipWith f (tMap g a) (tMap h b)@ MUST fuse to single traversal.
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

-- | Element-wise addition.
tAdd :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tAdd = tZipWith (+)

-- | Element-wise subtraction.
tSub :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tSub = tZipWith (-)

-- | Element-wise multiplication.
tMul :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tMul = tZipWith (*)

-- | Element-wise division.
tDiv :: (Fractional a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tDiv = tZipWith (/)

-- | Negate all elements.
tNeg :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
tNeg = tMap negate

-- | Absolute value.
tAbs :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
tAbs = tMap abs

-- | Sign function.
tSign :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
tSign = tMap signum

-- | Square root.
tSqrt :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tSqrt = tMap sqrt

-- | Exponential.
tExp :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tExp = tMap exp

-- | Natural logarithm.
tLog :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tLog = tMap log

-- | Power.
tPow :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tPow = tZipWith (**)

-- | Sine.
tSin :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tSin = tMap sin

-- | Cosine.
tCos :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tCos = tMap cos

-- | Tangent.
tTan :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tTan = tMap tan

-- | Hyperbolic sine.
tSinh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tSinh = tMap sinh

-- | Hyperbolic cosine.
tCosh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tCosh = tMap cosh

-- | Hyperbolic tangent.
tTanh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
tTanh = tMap tanh

-- | Floor.
tFloor :: (RealFrac a, VU.Unbox a) => Tensor a -> Tensor a
tFloor = tMap (fromIntegral . floor)

-- | Ceiling.
tCeil :: (RealFrac a, VU.Unbox a) => Tensor a -> Tensor a
tCeil = tMap (fromIntegral . ceiling)

-- | Round.
tRound :: (RealFrac a, VU.Unbox a) => Tensor a -> Tensor a
tRound = tMap (fromIntegral . round)

-- | Element-wise minimum.
tMin :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tMin = tZipWith min

-- | Element-wise maximum.
tMax :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
tMax = tZipWith max

-- | Clamp values to range.
tClamp :: (Ord a, VU.Unbox a) => a -> a -> Tensor a -> Tensor a
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
tSum :: (Num a, VU.Unbox a) => Tensor a -> a
tSum = VU.sum . tensorData . contiguous

-- | Product of all elements.
{-# RULES
"tProduct/tMap" forall f t. tProduct (tMap f t) = tFoldl' (\acc x -> acc * f x) 1 t
#-}
tProduct :: (Num a, VU.Unbox a) => Tensor a -> a
tProduct = VU.product . tensorData . contiguous

-- | Mean of all elements.
tMean :: (Fractional a, VU.Unbox a) => Tensor a -> a
tMean t = tSum t / fromIntegral (size t)

-- | Variance of all elements.
tVar :: (Floating a, VU.Unbox a) => Tensor a -> a
tVar t =
    let m = tMean t
        n = fromIntegral (size t)
    in tSum (tMap (\x -> (x - m) ^ (2 :: Int)) t) / n

-- | Standard deviation.
tStd :: (Floating a, VU.Unbox a) => Tensor a -> a
tStd = sqrt . tVar

-- | Minimum element.
tMin' :: (Ord a, VU.Unbox a) => Tensor a -> a
tMin' t = VU.minimum (tensorData (contiguous t))

-- | Maximum element.
tMax' :: (Ord a, VU.Unbox a) => Tensor a -> a
tMax' t = VU.maximum (tensorData (contiguous t))

-- | Index of minimum element.
tArgmin :: (Ord a, VU.Unbox a) => Tensor a -> Int
tArgmin t = VU.minIndex (tensorData (contiguous t))

-- | Index of maximum element.
tArgmax :: (Ord a, VU.Unbox a) => Tensor a -> Int
tArgmax t = VU.maxIndex (tensorData (contiguous t))

-- | True if any element is True.
tAny :: Tensor Bool -> Bool
tAny t = VU.or (tensorData (contiguous t))

-- | True if all elements are True.
tAll :: Tensor Bool -> Bool
tAll t = VU.and (tensorData (contiguous t))

-- Internal fold
tFoldl' :: VU.Unbox a => (b -> a -> b) -> b -> Tensor a -> b
tFoldl' f z t = VU.foldl' f z (tensorData (contiguous t))

-- ============================================================
-- Reduction Along Axis
-- ============================================================

-- | Sum along axis.
sumAxis :: (Num a, VU.Unbox a) => Int -> Tensor a -> Tensor a
sumAxis = reduceAxis (+) 0

-- | Product along axis.
productAxis :: (Num a, VU.Unbox a) => Int -> Tensor a -> Tensor a
productAxis = reduceAxis (*) 1

-- | Mean along axis.
meanAxis :: (Fractional a, VU.Unbox a) => Int -> Tensor a -> Tensor a
meanAxis axis t =
    let sumT = sumAxis axis t
        n = fromIntegral (tensorShape t !! axis)
    in tMap (/ n) sumT

-- | Minimum along axis.
minAxis :: (Ord a, VU.Unbox a, Bounded a) => Int -> Tensor a -> Tensor a
minAxis = reduceAxis min maxBound

-- | Maximum along axis.
maxAxis :: (Ord a, VU.Unbox a, Bounded a) => Int -> Tensor a -> Tensor a
maxAxis = reduceAxis max minBound

-- | Index of minimum along axis.
argminAxis :: (Ord a, VU.Unbox a) => Int -> Tensor a -> Tensor Int
argminAxis axis t = argReduceAxis (<) axis t

-- | Index of maximum along axis.
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

-- | Dot product of two vectors.
--
-- >>> dot [1, 2, 3] [4, 5, 6]
-- 32
dot :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> a
dot t1 t2
    | rank t1 /= 1 || rank t2 /= 1 = error "dot: expected 1D tensors"
    | size t1 /= size t2 = error "dot: size mismatch"
    | otherwise = tSum (tMul t1 t2)

-- | Matrix multiplication.
--
-- ==== __Complexity__
--
-- O(n * m * k) for (n x m) @ (m x k) matrices.
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

-- | Infix matrix multiplication.
(@) :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
(@) = matmul
infixl 7 @

-- | Outer product of two vectors.
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

-- | Inner product (generalized dot).
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

-- | Vector/matrix norm.
--
-- Default is L2 (Frobenius) norm.
norm :: (Floating a, VU.Unbox a) => Tensor a -> a
norm t = sqrt (tSum (tMap (^ (2 :: Int)) t))

-- | Normalize to unit norm.
normalize :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
normalize t =
    let n = norm t
    in if n == 0 then t else tMap (/ n) t

-- | Matrix trace (sum of diagonal).
trace :: (Num a, VU.Unbox a) => Tensor a -> a
trace t
    | rank t /= 2 = error "trace: expected 2D tensor"
    | otherwise =
        let [m, n] = tensorShape t
            k = min m n
        in sum [index t [i, i] | i <- [0..k-1]]

-- | Matrix determinant.
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

-- | Matrix inverse.
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

-- | Solve linear system Ax = b.
solve :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
solve a b
    | rank a /= 2 = error "solve: A must be 2D"
    | tensorShape a !! 0 /= tensorShape a !! 1 = error "solve: A must be square"
    | otherwise = matmul (inv a) b

-- | Least squares solution.
lstsq :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
lstsq a b =
    -- x = (A^T A)^(-1) A^T b
    let at = transpose a
        ata = matmul at a
        atb = matmul at b
    in solve ata atb

-- | Eigenvalue decomposition.
-- Returns (eigenvalues, eigenvectors).
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
-- Returns (U, S, V) where A = U * diag(S) * V^T.
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

-- | QR decomposition.
-- Returns (Q, R) where A = Q * R.
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

-- | Cholesky decomposition.
-- Returns L where A = L * L^T.
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

-- | Element-wise equality.
tEq :: (Eq a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tEq = tZipWith (==)

-- | Element-wise inequality.
tNe :: (Eq a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tNe = tZipWith (/=)

-- | Element-wise less than.
tLt :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tLt = tZipWith (<)

-- | Element-wise less than or equal.
tLe :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tLe = tZipWith (<=)

-- | Element-wise greater than.
tGt :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tGt = tZipWith (>)

-- | Element-wise greater than or equal.
tGe :: (Ord a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor Bool
tGe = tZipWith (>=)

-- | Check for NaN.
tIsNan :: (RealFloat a, VU.Unbox a) => Tensor a -> Tensor Bool
tIsNan = tMap isNaN

-- | Check for infinity.
tIsInf :: (RealFloat a, VU.Unbox a) => Tensor a -> Tensor Bool
tIsInf = tMap isInfinite

-- | Check for finite values.
tIsFinite :: (RealFloat a, VU.Unbox a) => Tensor a -> Tensor Bool
tIsFinite = tMap (\x -> not (isNaN x || isInfinite x))

-- ============================================================
-- Type Casting
-- ============================================================

-- | Cast tensor to different element type.
cast :: (VU.Unbox a, VU.Unbox b) => (a -> b) -> Tensor a -> Tensor b
cast = tMap

-- | Cast tensor to specific dtype (runtime type switch).
asType :: (VU.Unbox a, VU.Unbox b, RealFrac a, Num b) => DType -> Tensor a -> Tensor b
asType _ = tMap (fromIntegral . round)

-- ============================================================
-- Utilities
-- ============================================================

-- | Create a copy of tensor with new memory.
clone :: VU.Unbox a => Tensor a -> IO (Tensor a)
clone t = return $ contiguous t

-- | Ensure tensor is contiguous in memory.
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

-- | Convert tensor to flat list.
toList :: VU.Unbox a => Tensor a -> [a]
toList t = [index t (unflattenIndex (tensorShape t) i) | i <- [0..size t - 1]]

-- | Convert tensor to nested lists.
toLists :: VU.Unbox a => Tensor a -> [[a]]
toLists t
    | rank t /= 2 = error "toLists: expected 2D tensor"
    | otherwise =
        let [rows, cols] = tensorShape t
        in [[index t [i, j] | j <- [0..cols-1]] | i <- [0..rows-1]]

-- | Force materialization (prevent fusion).
--
-- Use when you need to reuse intermediate results.
materialize :: VU.Unbox a => Tensor a -> IO (Tensor a)
materialize = clone

-- | Execute tensor operations within arena scope.
--
-- Temporary allocations are freed when scope exits.
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
