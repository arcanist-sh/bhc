-- |
-- Module      : BHC.Numeric.Tensor.Shaped
-- Description : Type-level shape-indexed tensors
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : experimental
--
-- Tensors with compile-time shape checking. Dimension errors are caught
-- at compile time, not runtime.
--
-- = Key Differentiator
--
-- Unlike PyTorch/NumPy where shape errors are runtime exceptions:
--
-- @
-- # Python (runtime error)
-- a = np.zeros((3, 4))
-- b = np.zeros((5, 6))
-- a @ b  # RuntimeError: shapes not aligned
-- @
--
-- BHC catches this at compile time:
--
-- @
-- a :: Tensor '[3, 4] Float
-- b :: Tensor '[5, 6] Float
-- a `matmul` b  -- Compile error: 4 /= 5
-- @
--
-- = Usage
--
-- @
-- {-# LANGUAGE DataKinds #-}
-- {-# LANGUAGE TypeApplications #-}
--
-- import BHC.Numeric.Tensor.Shaped
--
-- -- Shape in type
-- x :: Tensor '[1024, 768] Float
-- x = zeros
--
-- -- Operations verify shapes at compile time
-- y :: Tensor '[768, 512] Float
-- y = ones
--
-- z :: Tensor '[1024, 512] Float
-- z = x `matmul` y  -- OK: 768 == 768
--
-- -- This would be a compile error:
-- -- bad = x `matmul` x  -- Error: 768 /= 1024
-- @

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ConstraintKinds #-}

module BHC.Numeric.Tensor.Shaped (
    -- * Shape-Indexed Tensor
    Tensor(..),

    -- * Shape Type Families
    type (++),
    Product,
    Broadcast,
    BroadcastValid,

    -- * Shape Classes
    KnownShape(..),
    SameProduct,

    -- * Construction
    zeros, ones, full,
    fromList,
    eye,

    -- * Shape Operations
    reshape,
    transpose,
    flatten,
    unsqueeze,
    squeeze,

    -- * Broadcast
    broadcast,

    -- * Element-wise Operations (MUST FUSE)
    tMap, tZipWith,
    tAdd, tSub, tMul, tDiv,
    tNeg, tAbs, tSqrt, tExp, tLog,

    -- * Reductions
    tSum, tMean,

    -- * Linear Algebra
    dot,
    matmul,
    matvec,

    -- * Dynamic Escape Hatch
    DynTensor(..),
    toDynamic,
    fromDynamic,
    withShape,

    -- * Utilities
    shape,
    shapeVal,
    toList,
) where

import GHC.TypeLits
import Data.Kind (Type, Constraint)
import Data.Proxy (Proxy(..))
import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Unboxed.Mutable as VUM
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad (forM_)

-- ============================================================
-- Type-Level Shape Utilities
-- ============================================================

-- | Append type-level lists
type family (xs :: [Nat]) ++ (ys :: [Nat]) :: [Nat] where
    '[] ++ ys = ys
    (x ': xs) ++ ys = x ': (xs ++ ys)

-- | Product of type-level list
type family Product (xs :: [Nat]) :: Nat where
    Product '[] = 1
    Product (x ': xs) = x * Product xs

-- | Compute broadcast result shape
--
-- Broadcasting rules (NumPy-compatible):
-- 1. Pad shorter shape with 1s on the left
-- 2. Dimensions must be equal or one of them must be 1
-- 3. Result dimension is the maximum
type family Broadcast (a :: [Nat]) (b :: [Nat]) :: [Nat] where
    Broadcast '[] b = b
    Broadcast a '[] = a
    Broadcast (1 ': as) (b ': bs) = b ': Broadcast as bs
    Broadcast (a ': as) (1 ': bs) = a ': Broadcast as bs
    Broadcast (a ': as) (a ': bs) = a ': Broadcast as bs
    -- If dimensions don't match and neither is 1, this won't reduce
    -- causing a compile error

-- | Constraint that broadcast is valid
type family BroadcastValid (a :: [Nat]) (b :: [Nat]) :: Constraint where
    BroadcastValid '[] _ = ()
    BroadcastValid _ '[] = ()
    BroadcastValid (1 ': as) (_ ': bs) = BroadcastValid as bs
    BroadcastValid (_ ': as) (1 ': bs) = BroadcastValid as bs
    BroadcastValid (a ': as) (a ': bs) = BroadcastValid as bs

-- | Class for shapes known at compile time
class KnownShape (shape :: [Nat]) where
    -- | Get the shape as a value-level list
    shapeVal' :: Proxy shape -> [Int]

    -- | Total number of elements
    sizeVal :: Proxy shape -> Int

instance KnownShape '[] where
    shapeVal' _ = []
    sizeVal _ = 1

instance (KnownNat n, KnownShape ns) => KnownShape (n ': ns) where
    shapeVal' _ = fromIntegral (natVal (Proxy @n)) : shapeVal' (Proxy @ns)
    sizeVal _ = fromIntegral (natVal (Proxy @n)) * sizeVal (Proxy @ns)

-- | Constraint for reshape: products must be equal
type SameProduct s1 s2 = (Product s1 ~ Product s2)

-- ============================================================
-- Shape-Indexed Tensor
-- ============================================================

-- | Tensor with shape tracked at type level.
--
-- The shape @s@ is a type-level list of natural numbers.
-- Operations verify shape compatibility at compile time.
--
-- @
-- Tensor '[3, 4] Float  -- 3x4 matrix of floats
-- Tensor '[10] Double   -- 10-element vector of doubles
-- Tensor '[] Float      -- scalar
-- @
data Tensor (shape :: [Nat]) a where
    Tensor :: VU.Unbox a
           => { tensorData :: !(VU.Vector a)
              }
           -> Tensor shape a

instance (KnownShape shape, Show a, VU.Unbox a) => Show (Tensor shape a) where
    show t = "Tensor " ++ show (shapeVal t) ++ " " ++ show (toList t)

instance (KnownShape shape, Eq a, VU.Unbox a) => Eq (Tensor shape a) where
    t1 == t2 = tensorData t1 == tensorData t2

-- ============================================================
-- Construction
-- ============================================================

-- | /O(n)/. Create tensor filled with zeros.
--
-- The shape is inferred from the type annotation.
--
-- @
-- x :: Tensor '[2, 3] Float
-- x = zeros  -- [[0,0,0],[0,0,0]]
-- @
zeros :: forall shape a. (KnownShape shape, Num a, VU.Unbox a) => Tensor shape a
zeros = Tensor (VU.replicate (sizeVal (Proxy @shape)) 0)

-- | /O(n)/. Create tensor filled with ones.
--
-- @
-- x :: Tensor '[3] Float
-- x = ones  -- [1, 1, 1]
-- @
ones :: forall shape a. (KnownShape shape, Num a, VU.Unbox a) => Tensor shape a
ones = Tensor (VU.replicate (sizeVal (Proxy @shape)) 1)

-- | /O(n)/. Create tensor filled with a constant value.
--
-- @
-- x :: Tensor '[2, 2] Float
-- x = full 3.14  -- [[3.14, 3.14], [3.14, 3.14]]
-- @
full :: forall shape a. (KnownShape shape, VU.Unbox a) => a -> Tensor shape a
full x = Tensor (VU.replicate (sizeVal (Proxy @shape)) x)

-- | /O(n)/. Create tensor from a flat list.
--
-- List length must match the shape product (checked at runtime).
--
-- @
-- x :: Tensor '[2, 3] Int
-- x = fromList [1,2,3,4,5,6]  -- [[1,2,3],[4,5,6]]
-- @
--
-- __Warning__: Throws error if list length doesn't match shape product.
fromList :: forall shape a. (KnownShape shape, VU.Unbox a) => [a] -> Tensor shape a
fromList xs
    | length xs /= sizeVal (Proxy @shape) =
        error $ "fromList: expected " ++ show (sizeVal (Proxy @shape))
             ++ " elements, got " ++ show (length xs)
    | otherwise = Tensor (VU.fromList xs)

-- | /O(n²)/. Create an n×n identity matrix.
--
-- @
-- i :: Tensor '[3, 3] Float
-- i = eye  -- [[1,0,0],[0,1,0],[0,0,1]]
-- @
eye :: forall n a. (KnownNat n, Num a, VU.Unbox a) => Tensor '[n, n] a
eye = unsafePerformIO $ do
    let n = fromIntegral (natVal (Proxy @n))
    mv <- VUM.replicate (n * n) 0
    forM_ [0..n-1] $ \i ->
        VUM.write mv (i * n + i) 1
    v <- VU.freeze mv
    return (Tensor v)

-- ============================================================
-- Shape Operations
-- ============================================================

-- | /O(1)/. Get shape as value-level list.
--
-- @
-- x :: Tensor '[2, 3, 4] Float
-- shape x  -- [2, 3, 4]
-- @
shape :: forall shape a. KnownShape shape => Tensor shape a -> [Int]
shape _ = shapeVal' (Proxy @shape)

-- | /O(1)/. Get shape as value (alias for 'shape').
shapeVal :: forall shape a. KnownShape shape => Tensor shape a -> [Int]
shapeVal = shape

-- | /O(1)/. Reshape tensor to a new shape.
--
-- The new shape must have the same total element count.
-- This is verified at compile time via the 'SameProduct' constraint.
--
-- @
-- x :: Tensor '[2, 6] Float
-- x = reshape (ones :: Tensor '[3, 4] Float)  -- 3*4 == 2*6 ✓
-- @
--
-- This is a zero-copy operation (view only).
reshape :: forall s1 s2 a. (SameProduct s1 s2, VU.Unbox a)
        => Tensor s1 a -> Tensor s2 a
reshape (Tensor v) = Tensor v

-- | /O(n)/. Transpose a 2D tensor (swap rows and columns).
--
-- @
-- x :: Tensor '[3, 4] Float
-- y :: Tensor '[4, 3] Float
-- y = transpose x
-- @
transpose :: forall m n a. (KnownNat m, KnownNat n, VU.Unbox a)
          => Tensor '[m, n] a -> Tensor '[n, m] a
transpose (Tensor v) = unsafePerformIO $ do
    let m = fromIntegral (natVal (Proxy @m))
        n = fromIntegral (natVal (Proxy @n))
    mv <- VUM.new (m * n)
    forM_ [0..m-1] $ \i ->
        forM_ [0..n-1] $ \j -> do
            let srcIdx = i * n + j
                dstIdx = j * m + i
            VUM.write mv dstIdx (v VU.! srcIdx)
    Tensor <$> VU.freeze mv

-- | /O(1)/. Flatten tensor to 1D.
--
-- @
-- x :: Tensor '[2, 3] Float
-- y :: Tensor '[6] Float
-- y = flatten x
-- @
--
-- Zero-copy operation (view only).
flatten :: forall shape a. (KnownShape shape, VU.Unbox a)
        => Tensor shape a -> Tensor '[Product shape] a
flatten (Tensor v) = Tensor v

-- | /O(1)/. Add a dimension of size 1 at the specified position.
--
-- Position is specified via TypeApplications.
--
-- @
-- x :: Tensor '[3, 4] Float
-- y :: Tensor '[1, 3, 4] Float
-- y = unsqueeze @0 x
-- @
unsqueeze :: forall dim shape a. VU.Unbox a
          => Tensor shape a -> Tensor (InsertAt dim 1 shape) a
unsqueeze (Tensor v) = Tensor v

-- | /O(1)/. Remove all dimensions of size 1.
--
-- Requires explicit result type annotation.
--
-- @
-- x :: Tensor '[1, 3, 1, 4] Float
-- y :: Tensor '[3, 4] Float
-- y = squeeze x
-- @
squeeze :: forall shape result a. (SameProduct shape result, VU.Unbox a)
        => Tensor shape a -> Tensor result a
squeeze (Tensor v) = Tensor v

-- Type family for inserting at position
type family InsertAt (n :: Nat) (x :: Nat) (xs :: [Nat]) :: [Nat] where
    InsertAt 0 x xs = x ': xs
    InsertAt n x (y ': ys) = y ': InsertAt (n - 1) x ys

-- ============================================================
-- Broadcasting
-- ============================================================
--
-- NumPy-compatible broadcasting semantics. Dimensions are compatible if:
-- - They are equal, OR
-- - One of them is 1 (which gets broadcasted)

-- | /O(n)/. Broadcast tensor to a larger shape.
--
-- Broadcasting rules (NumPy-compatible):
-- 1. Pad shorter shape with 1s on the left
-- 2. Dimensions must be equal or one must be 1
-- 3. Result dimension is the maximum
--
-- @
-- x :: Tensor '[3] Float
-- y :: Tensor '[4, 3] Float
-- y = broadcast x  -- [3] broadcasts to [4, 3]
-- @
--
-- Shape compatibility is verified at compile time.
broadcast :: forall shape target a.
             ( KnownShape shape
             , KnownShape target
             , BroadcastValid shape target
             , Broadcast shape target ~ target
             , VU.Unbox a
             )
          => Tensor shape a -> Tensor target a
broadcast (Tensor v) = unsafePerformIO $ do
    let srcShape = shapeVal' (Proxy @shape)
        tgtShape = shapeVal' (Proxy @target)
        tgtSize = sizeVal (Proxy @target)
        -- Pad source shape with 1s
        paddedSrc = replicate (length tgtShape - length srcShape) 1 ++ srcShape
        srcStrides = computeStrides paddedSrc
        tgtStrides = computeStrides tgtShape
    mv <- VUM.new tgtSize
    forM_ [0..tgtSize-1] $ \i -> do
        let tgtIndices = unflattenIndex tgtShape i
            srcIndices = zipWith (\t s -> if s == 1 then 0 else t) tgtIndices paddedSrc
            srcIdx = sum (zipWith (*) srcIndices srcStrides)
        VUM.write mv i (v VU.! min srcIdx (VU.length v - 1))
    Tensor <$> VU.freeze mv

-- ============================================================
-- Element-wise Operations (MUST FUSE)
-- ============================================================
--
-- All element-wise operations are designed to fuse according
-- to H26-SPEC Section 8. Shape compatibility is verified at
-- compile time via the type system.

-- | /O(n)/. Map a function over tensor elements.
--
-- ==== __Fusion__
--
-- @tMap f (tMap g t)@ MUST fuse to @tMap (f . g) t@ (single traversal).
-- This is a guaranteed fusion pattern per H26-SPEC Section 8.1.
{-# RULES
"tMap/tMap" forall f g t. tMap f (tMap g t) = tMap (f . g) t
#-}
tMap :: (VU.Unbox a, VU.Unbox b) => (a -> b) -> Tensor shape a -> Tensor shape b
tMap f (Tensor v) = Tensor (VU.map f v)

-- | /O(n)/. Combine two tensors element-wise with a binary function.
--
-- Shapes must match exactly (verified at compile time).
--
-- ==== __Fusion__
--
-- @tZipWith f (tMap g a) (tMap h b)@ fuses to a single traversal.
{-# RULES
"tZipWith/tMap/tMap" forall f g h a b.
    tZipWith f (tMap g a) (tMap h b) = tZipWith (\x y -> f (g x) (h y)) a b
#-}
tZipWith :: (VU.Unbox a, VU.Unbox b, VU.Unbox c)
         => (a -> b -> c) -> Tensor shape a -> Tensor shape b -> Tensor shape c
tZipWith f (Tensor v1) (Tensor v2) = Tensor (VU.zipWith f v1 v2)

-- | /O(n)/. Element-wise addition.
tAdd :: (Num a, VU.Unbox a) => Tensor shape a -> Tensor shape a -> Tensor shape a
tAdd = tZipWith (+)

-- | /O(n)/. Element-wise subtraction.
tSub :: (Num a, VU.Unbox a) => Tensor shape a -> Tensor shape a -> Tensor shape a
tSub = tZipWith (-)

-- | /O(n)/. Element-wise multiplication (Hadamard product).
tMul :: (Num a, VU.Unbox a) => Tensor shape a -> Tensor shape a -> Tensor shape a
tMul = tZipWith (*)

-- | /O(n)/. Element-wise division.
tDiv :: (Fractional a, VU.Unbox a) => Tensor shape a -> Tensor shape a -> Tensor shape a
tDiv = tZipWith (/)

-- | /O(n)/. Negate all elements.
tNeg :: (Num a, VU.Unbox a) => Tensor shape a -> Tensor shape a
tNeg = tMap negate

-- | /O(n)/. Absolute value of all elements.
tAbs :: (Num a, VU.Unbox a) => Tensor shape a -> Tensor shape a
tAbs = tMap abs

-- | /O(n)/. Square root of all elements.
tSqrt :: (Floating a, VU.Unbox a) => Tensor shape a -> Tensor shape a
tSqrt = tMap sqrt

-- | /O(n)/. Exponential (e^x) of all elements.
tExp :: (Floating a, VU.Unbox a) => Tensor shape a -> Tensor shape a
tExp = tMap exp

-- | /O(n)/. Natural logarithm of all elements.
tLog :: (Floating a, VU.Unbox a) => Tensor shape a -> Tensor shape a
tLog = tMap log

-- ============================================================
-- Reductions
-- ============================================================
--
-- Reduce tensor to a scalar value. All reductions fuse with
-- preceding map operations.

-- | /O(n)/. Sum of all tensor elements.
--
-- ==== __Fusion__
--
-- @tSum (tMap f t)@ MUST fuse to a single traversal with no
-- intermediate allocation. This is a guaranteed fusion pattern.
{-# RULES
"tSum/tMap" forall f t. tSum (tMap f t) = VU.foldl' (\acc x -> acc + f x) 0 (tensorData t)
#-}
tSum :: (Num a, VU.Unbox a) => Tensor shape a -> a
tSum (Tensor v) = VU.sum v

-- | /O(n)/. Arithmetic mean of all tensor elements.
tMean :: forall shape a. (KnownShape shape, Fractional a, VU.Unbox a)
      => Tensor shape a -> a
tMean t = tSum t / fromIntegral (sizeVal (Proxy @shape))

-- ============================================================
-- Linear Algebra
-- ============================================================
--
-- Type-safe linear algebra operations. Dimension compatibility
-- is verified at compile time via the type system.

-- | /O(n)/. Dot product (inner product) of two vectors.
--
-- Vector lengths must match (verified at compile time).
--
-- @
-- x :: Tensor '[100] Float
-- y :: Tensor '[100] Float
-- s = dot x y  -- scalar result
-- @
dot :: (Num a, VU.Unbox a) => Tensor '[n] a -> Tensor '[n] a -> a
dot (Tensor v1) (Tensor v2) = VU.sum (VU.zipWith (*) v1 v2)

-- | /O(m·k·n)/. Matrix multiplication.
--
-- The inner dimension @k@ must match (verified at compile time).
--
-- @
-- a :: Tensor '[m, k] Float
-- b :: Tensor '[k, n] Float
-- c :: Tensor '[m, n] Float
-- c = a \`matmul\` b
-- @
--
-- Dimension mismatch is a compile error:
--
-- @
-- -- This would not compile: 768 /= 512
-- bad = (ones :: Tensor '[1024, 768] Float)
--       \`matmul\` (ones :: Tensor '[512, 256] Float)
-- @
matmul :: forall m k n a. (KnownNat m, KnownNat k, KnownNat n, Num a, VU.Unbox a)
       => Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a
matmul (Tensor v1) (Tensor v2) = unsafePerformIO $ do
    let m = fromIntegral (natVal (Proxy @m))
        k = fromIntegral (natVal (Proxy @k))
        n = fromIntegral (natVal (Proxy @n))
    mv <- VUM.new (m * n)
    forM_ [0..m-1] $ \i ->
        forM_ [0..n-1] $ \j -> do
            let val = sum [v1 VU.! (i * k + l) * v2 VU.! (l * n + j) | l <- [0..k-1]]
            VUM.write mv (i * n + j) val
    Tensor <$> VU.freeze mv

-- | /O(m·n)/. Matrix-vector multiplication.
--
-- @
-- a :: Tensor '[3, 4] Float  -- 3×4 matrix
-- x :: Tensor '[4] Float     -- 4-element vector
-- y :: Tensor '[3] Float     -- 3-element result
-- y = matvec a x
-- @
matvec :: forall m n a. (KnownNat m, KnownNat n, Num a, VU.Unbox a)
       => Tensor '[m, n] a -> Tensor '[n] a -> Tensor '[m] a
matvec (Tensor mat) (Tensor vec) = unsafePerformIO $ do
    let m = fromIntegral (natVal (Proxy @m))
        n = fromIntegral (natVal (Proxy @n))
    mv <- VUM.new m
    forM_ [0..m-1] $ \i -> do
        let val = sum [mat VU.! (i * n + j) * vec VU.! j | j <- [0..n-1]]
        VUM.write mv i val
    Tensor <$> VU.freeze mv

-- ============================================================
-- Dynamic Escape Hatch
-- ============================================================
--
-- For interop with external systems or when shapes are unknown
-- at compile time.

-- | Dynamic tensor with runtime shape.
--
-- Use when shapes are not known at compile time, such as when
-- reading data from files or receiving from external APIs.
data DynTensor a = DynTensor
    { dynData  :: !(VU.Vector a)
    , dynShape :: ![Int]
    } deriving (Show, Eq)

-- | /O(1)/. Convert a shaped tensor to a dynamic tensor.
--
-- Loses compile-time shape checking but allows runtime flexibility.
toDynamic :: KnownShape shape => Tensor shape a -> DynTensor a
toDynamic t@(Tensor v) = DynTensor v (shapeVal t)

-- | /O(1)/. Attempt to convert a dynamic tensor to a shaped tensor.
--
-- Returns @Nothing@ if the runtime shape doesn't match the expected
-- compile-time shape.
--
-- @
-- dyn :: DynTensor Float  -- shape [2, 3] at runtime
--
-- case fromDynamic @'[2, 3] dyn of
--   Just t  -> use t  -- Shape verified
--   Nothing -> error "Shape mismatch"
-- @
fromDynamic :: forall shape a. (KnownShape shape, VU.Unbox a)
            => DynTensor a -> Maybe (Tensor shape a)
fromDynamic (DynTensor v sh)
    | sh == shapeVal' (Proxy @shape) = Just (Tensor v)
    | otherwise = Nothing

-- | Apply a function to a dynamic tensor if it matches the expected shape.
--
-- Combines 'fromDynamic' and function application.
withShape :: forall shape a r. (KnownShape shape, VU.Unbox a)
          => DynTensor a
          -> (Tensor shape a -> r)
          -> Maybe r
withShape dyn f = f <$> fromDynamic dyn

-- ============================================================
-- Utilities
-- ============================================================

-- | /O(n)/. Convert tensor to a flat list (row-major order).
--
-- @
-- x :: Tensor '[2, 3] Int
-- toList x  -- [1,2,3,4,5,6]
-- @
toList :: VU.Unbox a => Tensor shape a -> [a]
toList (Tensor v) = VU.toList v

-- ============================================================
-- Internal Helpers
-- ============================================================

computeStrides :: [Int] -> [Int]
computeStrides [] = []
computeStrides sh = scanr (*) 1 (tail sh)

unflattenIndex :: [Int] -> Int -> [Int]
unflattenIndex sh idx = go (reverse sh) idx []
  where
    go [] _ acc = acc
    go (d:ds) i acc =
        let (q, r) = i `divMod` d
        in go ds q (r : acc)
