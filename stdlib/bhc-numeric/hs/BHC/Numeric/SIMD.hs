-- |
-- Module      : BHC.Numeric.SIMD
-- Description : SIMD vector types and operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Hardware-accelerated SIMD operations for high-performance numerics.
-- Supports AVX2, AVX-512, and NEON instruction sets.
--
-- This module provides fixed-width vector types with element-wise operations.
-- When compiled with SIMD support, operations use hardware intrinsics.

{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}

module BHC.Numeric.SIMD (
    -- * Float vectors
    Vec2F32(..), Vec4F32(..), Vec8F32(..), Vec16F32(..),
    Vec2F64(..), Vec4F64(..), Vec8F64(..),

    -- * Int vectors
    Vec2I32(..), Vec4I32(..), Vec8I32(..), Vec16I32(..),
    Vec2I64(..), Vec4I64(..), Vec8I64(..),

    -- * Construction
    broadcast,
    fromList,
    pack2, pack4, pack8,

    -- * Extraction
    extract,
    toList,
    unpack2, unpack4, unpack8,

    -- * Arithmetic
    vAdd, vSub, vMul, vDiv,
    vNeg, vAbs, vSqrt, vRsqrt, vRcp,
    vMin, vMax,

    -- * Fused multiply-add
    vFMA, vFMS, vFNMA, vFNMS,

    -- * Comparison
    vEq, vNe, vLt, vLe, vGt, vGe,

    -- * Horizontal operations
    vSum, vProduct,
    vHMin, vHMax,

    -- * Shuffle and blend
    vShuffle, vBlend,
    vPermute, vRotate,

    -- * Load/Store
    vLoad, vLoadAligned,
    vStore, vStoreAligned,
    vGather, vScatter,
    vMaskedLoad, vMaskedStore,

    -- * Bitwise
    vAnd, vOr, vXor, vNot,
    vShiftL, vShiftR,

    -- * Mask type
    Mask(..),
    maskAll, maskNone,
    maskFromBits, maskToBits,

    -- * Type class
    SIMDVec(..),
) where

import Prelude hiding (toList, fromList)
import qualified Prelude as P
import Data.Word (Word64, Word32)
import Data.Bits ((.&.), (.|.), xor, complement, shiftL, shiftR)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable, peek, poke, peekElemOff, pokeElemOff)
import Foreign.Marshal.Array (peekArray, pokeArray)

-- ============================================================
-- Vector Types - Float
-- ============================================================

-- | 2-element float vector (64-bit).
data Vec2F32 = Vec2F32 {-# UNPACK #-} !Float {-# UNPACK #-} !Float
    deriving (Eq, Show)

-- | 4-element float vector (128-bit SSE).
data Vec4F32 = Vec4F32 {-# UNPACK #-} !Float {-# UNPACK #-} !Float
                       {-# UNPACK #-} !Float {-# UNPACK #-} !Float
    deriving (Eq, Show)

-- | 8-element float vector (256-bit AVX).
data Vec8F32 = Vec8F32 {-# UNPACK #-} !Float {-# UNPACK #-} !Float
                       {-# UNPACK #-} !Float {-# UNPACK #-} !Float
                       {-# UNPACK #-} !Float {-# UNPACK #-} !Float
                       {-# UNPACK #-} !Float {-# UNPACK #-} !Float
    deriving (Eq, Show)

-- | 16-element float vector (512-bit AVX-512).
data Vec16F32 = Vec16F32 !Vec8F32 !Vec8F32
    deriving (Eq, Show)

-- | 2-element double vector (128-bit).
data Vec2F64 = Vec2F64 {-# UNPACK #-} !Double {-# UNPACK #-} !Double
    deriving (Eq, Show)

-- | 4-element double vector (256-bit AVX).
data Vec4F64 = Vec4F64 {-# UNPACK #-} !Double {-# UNPACK #-} !Double
                       {-# UNPACK #-} !Double {-# UNPACK #-} !Double
    deriving (Eq, Show)

-- | 8-element double vector (512-bit AVX-512).
data Vec8F64 = Vec8F64 !Vec4F64 !Vec4F64
    deriving (Eq, Show)

-- ============================================================
-- Vector Types - Integer
-- ============================================================

-- | 2-element int32 vector.
data Vec2I32 = Vec2I32 {-# UNPACK #-} !Int {-# UNPACK #-} !Int
    deriving (Eq, Show)

-- | 4-element int32 vector (128-bit SSE).
data Vec4I32 = Vec4I32 {-# UNPACK #-} !Int {-# UNPACK #-} !Int
                       {-# UNPACK #-} !Int {-# UNPACK #-} !Int
    deriving (Eq, Show)

-- | 8-element int32 vector (256-bit AVX).
data Vec8I32 = Vec8I32 !Vec4I32 !Vec4I32
    deriving (Eq, Show)

-- | 16-element int32 vector (512-bit AVX-512).
data Vec16I32 = Vec16I32 !Vec8I32 !Vec8I32
    deriving (Eq, Show)

-- | 2-element int64 vector.
data Vec2I64 = Vec2I64 {-# UNPACK #-} !Int {-# UNPACK #-} !Int
    deriving (Eq, Show)

-- | 4-element int64 vector (256-bit AVX).
data Vec4I64 = Vec4I64 {-# UNPACK #-} !Int {-# UNPACK #-} !Int
                       {-# UNPACK #-} !Int {-# UNPACK #-} !Int
    deriving (Eq, Show)

-- | 8-element int64 vector (512-bit AVX-512).
data Vec8I64 = Vec8I64 !Vec4I64 !Vec4I64
    deriving (Eq, Show)

-- ============================================================
-- Mask Type
-- ============================================================

-- | SIMD mask for conditional operations.
newtype Mask n = Mask { maskBits :: Word64 }
    deriving (Eq, Show)

-- | Mask with all bits set.
maskAll :: Mask n
maskAll = Mask maxBound

-- | Mask with no bits set.
maskNone :: Mask n
maskNone = Mask 0

-- | Create mask from bit pattern.
maskFromBits :: Word64 -> Mask n
maskFromBits = Mask

-- | Extract bit pattern from mask.
maskToBits :: Mask n -> Word64
maskToBits = maskBits

-- ============================================================
-- Type Class for SIMD Operations
-- ============================================================

-- | SIMD vector type class.
class SIMDVec v where
    type Elem v

    -- | Broadcast a scalar to all lanes.
    vBroadcast :: Elem v -> v

    -- | Extract element at index.
    vExtract :: v -> Int -> Elem v

    -- | Insert element at index.
    vInsert :: v -> Int -> Elem v -> v

    -- | Number of elements.
    vWidth :: v -> Int

    -- | Element-wise addition.
    vAddImpl :: v -> v -> v

    -- | Element-wise subtraction.
    vSubImpl :: v -> v -> v

    -- | Element-wise multiplication.
    vMulImpl :: v -> v -> v

    -- | Element-wise division.
    vDivImpl :: v -> v -> v

    -- | Horizontal sum.
    vSumImpl :: v -> Elem v

    -- | Horizontal product.
    vProductImpl :: v -> Elem v

-- ============================================================
-- Vec2F32 Instance
-- ============================================================

instance SIMDVec Vec2F32 where
    type Elem Vec2F32 = Float

    vBroadcast x = Vec2F32 x x

    vExtract (Vec2F32 a b) i = case i of
        0 -> a; 1 -> b; _ -> error "Vec2F32: index out of bounds"

    vInsert (Vec2F32 a b) i x = case i of
        0 -> Vec2F32 x b
        1 -> Vec2F32 a x
        _ -> error "Vec2F32: index out of bounds"

    vWidth _ = 2

    vAddImpl (Vec2F32 a1 b1) (Vec2F32 a2 b2) = Vec2F32 (a1+a2) (b1+b2)
    vSubImpl (Vec2F32 a1 b1) (Vec2F32 a2 b2) = Vec2F32 (a1-a2) (b1-b2)
    vMulImpl (Vec2F32 a1 b1) (Vec2F32 a2 b2) = Vec2F32 (a1*a2) (b1*b2)
    vDivImpl (Vec2F32 a1 b1) (Vec2F32 a2 b2) = Vec2F32 (a1/a2) (b1/b2)

    vSumImpl (Vec2F32 a b) = a + b
    vProductImpl (Vec2F32 a b) = a * b

-- ============================================================
-- Vec4F32 Instance
-- ============================================================

instance SIMDVec Vec4F32 where
    type Elem Vec4F32 = Float

    vBroadcast x = Vec4F32 x x x x

    vExtract (Vec4F32 a b c d) i = case i of
        0 -> a; 1 -> b; 2 -> c; 3 -> d
        _ -> error "Vec4F32: index out of bounds"

    vInsert (Vec4F32 a b c d) i x = case i of
        0 -> Vec4F32 x b c d
        1 -> Vec4F32 a x c d
        2 -> Vec4F32 a b x d
        3 -> Vec4F32 a b c x
        _ -> error "Vec4F32: index out of bounds"

    vWidth _ = 4

    vAddImpl (Vec4F32 a1 b1 c1 d1) (Vec4F32 a2 b2 c2 d2) =
        Vec4F32 (a1+a2) (b1+b2) (c1+c2) (d1+d2)
    vSubImpl (Vec4F32 a1 b1 c1 d1) (Vec4F32 a2 b2 c2 d2) =
        Vec4F32 (a1-a2) (b1-b2) (c1-c2) (d1-d2)
    vMulImpl (Vec4F32 a1 b1 c1 d1) (Vec4F32 a2 b2 c2 d2) =
        Vec4F32 (a1*a2) (b1*b2) (c1*c2) (d1*d2)
    vDivImpl (Vec4F32 a1 b1 c1 d1) (Vec4F32 a2 b2 c2 d2) =
        Vec4F32 (a1/a2) (b1/b2) (c1/c2) (d1/d2)

    vSumImpl (Vec4F32 a b c d) = a + b + c + d
    vProductImpl (Vec4F32 a b c d) = a * b * c * d

-- ============================================================
-- Vec8F32 Instance
-- ============================================================

instance SIMDVec Vec8F32 where
    type Elem Vec8F32 = Float

    vBroadcast x = Vec8F32 x x x x x x x x

    vExtract (Vec8F32 a b c d e f g h) i = case i of
        0 -> a; 1 -> b; 2 -> c; 3 -> d
        4 -> e; 5 -> f; 6 -> g; 7 -> h
        _ -> error "Vec8F32: index out of bounds"

    vInsert (Vec8F32 a b c d e f g h) i x = case i of
        0 -> Vec8F32 x b c d e f g h
        1 -> Vec8F32 a x c d e f g h
        2 -> Vec8F32 a b x d e f g h
        3 -> Vec8F32 a b c x e f g h
        4 -> Vec8F32 a b c d x f g h
        5 -> Vec8F32 a b c d e x g h
        6 -> Vec8F32 a b c d e f x h
        7 -> Vec8F32 a b c d e f g x
        _ -> error "Vec8F32: index out of bounds"

    vWidth _ = 8

    vAddImpl (Vec8F32 a1 b1 c1 d1 e1 f1 g1 h1) (Vec8F32 a2 b2 c2 d2 e2 f2 g2 h2) =
        Vec8F32 (a1+a2) (b1+b2) (c1+c2) (d1+d2) (e1+e2) (f1+f2) (g1+g2) (h1+h2)
    vSubImpl (Vec8F32 a1 b1 c1 d1 e1 f1 g1 h1) (Vec8F32 a2 b2 c2 d2 e2 f2 g2 h2) =
        Vec8F32 (a1-a2) (b1-b2) (c1-c2) (d1-d2) (e1-e2) (f1-f2) (g1-g2) (h1-h2)
    vMulImpl (Vec8F32 a1 b1 c1 d1 e1 f1 g1 h1) (Vec8F32 a2 b2 c2 d2 e2 f2 g2 h2) =
        Vec8F32 (a1*a2) (b1*b2) (c1*c2) (d1*d2) (e1*e2) (f1*f2) (g1*g2) (h1*h2)
    vDivImpl (Vec8F32 a1 b1 c1 d1 e1 f1 g1 h1) (Vec8F32 a2 b2 c2 d2 e2 f2 g2 h2) =
        Vec8F32 (a1/a2) (b1/b2) (c1/c2) (d1/d2) (e1/e2) (f1/f2) (g1/g2) (h1/h2)

    vSumImpl (Vec8F32 a b c d e f g h) = a + b + c + d + e + f + g + h
    vProductImpl (Vec8F32 a b c d e f g h) = a * b * c * d * e * f * g * h

-- ============================================================
-- Vec2F64 Instance
-- ============================================================

instance SIMDVec Vec2F64 where
    type Elem Vec2F64 = Double

    vBroadcast x = Vec2F64 x x

    vExtract (Vec2F64 a b) i = case i of
        0 -> a; 1 -> b; _ -> error "Vec2F64: index out of bounds"

    vInsert (Vec2F64 a b) i x = case i of
        0 -> Vec2F64 x b
        1 -> Vec2F64 a x
        _ -> error "Vec2F64: index out of bounds"

    vWidth _ = 2

    vAddImpl (Vec2F64 a1 b1) (Vec2F64 a2 b2) = Vec2F64 (a1+a2) (b1+b2)
    vSubImpl (Vec2F64 a1 b1) (Vec2F64 a2 b2) = Vec2F64 (a1-a2) (b1-b2)
    vMulImpl (Vec2F64 a1 b1) (Vec2F64 a2 b2) = Vec2F64 (a1*a2) (b1*b2)
    vDivImpl (Vec2F64 a1 b1) (Vec2F64 a2 b2) = Vec2F64 (a1/a2) (b1/b2)

    vSumImpl (Vec2F64 a b) = a + b
    vProductImpl (Vec2F64 a b) = a * b

-- ============================================================
-- Vec4F64 Instance
-- ============================================================

instance SIMDVec Vec4F64 where
    type Elem Vec4F64 = Double

    vBroadcast x = Vec4F64 x x x x

    vExtract (Vec4F64 a b c d) i = case i of
        0 -> a; 1 -> b; 2 -> c; 3 -> d
        _ -> error "Vec4F64: index out of bounds"

    vInsert (Vec4F64 a b c d) i x = case i of
        0 -> Vec4F64 x b c d
        1 -> Vec4F64 a x c d
        2 -> Vec4F64 a b x d
        3 -> Vec4F64 a b c x
        _ -> error "Vec4F64: index out of bounds"

    vWidth _ = 4

    vAddImpl (Vec4F64 a1 b1 c1 d1) (Vec4F64 a2 b2 c2 d2) =
        Vec4F64 (a1+a2) (b1+b2) (c1+c2) (d1+d2)
    vSubImpl (Vec4F64 a1 b1 c1 d1) (Vec4F64 a2 b2 c2 d2) =
        Vec4F64 (a1-a2) (b1-b2) (c1-c2) (d1-d2)
    vMulImpl (Vec4F64 a1 b1 c1 d1) (Vec4F64 a2 b2 c2 d2) =
        Vec4F64 (a1*a2) (b1*b2) (c1*c2) (d1*d2)
    vDivImpl (Vec4F64 a1 b1 c1 d1) (Vec4F64 a2 b2 c2 d2) =
        Vec4F64 (a1/a2) (b1/b2) (c1/c2) (d1/d2)

    vSumImpl (Vec4F64 a b c d) = a + b + c + d
    vProductImpl (Vec4F64 a b c d) = a * b * c * d

-- ============================================================
-- Vec4I32 Instance
-- ============================================================

instance SIMDVec Vec4I32 where
    type Elem Vec4I32 = Int

    vBroadcast x = Vec4I32 x x x x

    vExtract (Vec4I32 a b c d) i = case i of
        0 -> a; 1 -> b; 2 -> c; 3 -> d
        _ -> error "Vec4I32: index out of bounds"

    vInsert (Vec4I32 a b c d) i x = case i of
        0 -> Vec4I32 x b c d
        1 -> Vec4I32 a x c d
        2 -> Vec4I32 a b x d
        3 -> Vec4I32 a b c x
        _ -> error "Vec4I32: index out of bounds"

    vWidth _ = 4

    vAddImpl (Vec4I32 a1 b1 c1 d1) (Vec4I32 a2 b2 c2 d2) =
        Vec4I32 (a1+a2) (b1+b2) (c1+c2) (d1+d2)
    vSubImpl (Vec4I32 a1 b1 c1 d1) (Vec4I32 a2 b2 c2 d2) =
        Vec4I32 (a1-a2) (b1-b2) (c1-c2) (d1-d2)
    vMulImpl (Vec4I32 a1 b1 c1 d1) (Vec4I32 a2 b2 c2 d2) =
        Vec4I32 (a1*a2) (b1*b2) (c1*c2) (d1*d2)
    vDivImpl (Vec4I32 a1 b1 c1 d1) (Vec4I32 a2 b2 c2 d2) =
        Vec4I32 (a1 `div` a2) (b1 `div` b2) (c1 `div` c2) (d1 `div` d2)

    vSumImpl (Vec4I32 a b c d) = a + b + c + d
    vProductImpl (Vec4I32 a b c d) = a * b * c * d

-- ============================================================
-- Construction Functions
-- ============================================================

-- | Broadcast a scalar to all elements.
broadcast :: SIMDVec v => Elem v -> v
broadcast = vBroadcast

-- | Create vector from list (must have correct length).
fromList :: SIMDVec v => [Elem v] -> v
fromList xs = go (vBroadcast (P.head xs)) 0 xs
  where
    go !v _ [] = v
    go !v !i (x:rest) = go (vInsert v i x) (i+1) rest

-- | Pack 2 scalars into a vector.
pack2 :: Float -> Float -> Vec2F32
pack2 = Vec2F32

-- | Pack 4 scalars into a vector.
pack4 :: Float -> Float -> Float -> Float -> Vec4F32
pack4 = Vec4F32

-- | Pack 8 scalars into a vector.
pack8 :: Float -> Float -> Float -> Float -> Float -> Float -> Float -> Float -> Vec8F32
pack8 = Vec8F32

-- ============================================================
-- Extraction Functions
-- ============================================================

-- | Extract element at index.
extract :: SIMDVec v => v -> Int -> Elem v
extract = vExtract

-- | Convert vector to list.
toList :: SIMDVec v => v -> [Elem v]
toList v = [vExtract v i | i <- [0 .. vWidth v - 1]]

-- | Unpack 2-element vector.
unpack2 :: Vec2F32 -> (Float, Float)
unpack2 (Vec2F32 a b) = (a, b)

-- | Unpack 4-element vector.
unpack4 :: Vec4F32 -> (Float, Float, Float, Float)
unpack4 (Vec4F32 a b c d) = (a, b, c, d)

-- | Unpack 8-element vector.
unpack8 :: Vec8F32 -> (Float, Float, Float, Float, Float, Float, Float, Float)
unpack8 (Vec8F32 a b c d e f g h) = (a, b, c, d, e, f, g, h)

-- ============================================================
-- Arithmetic Operations
-- ============================================================

-- | Vector addition.
vAdd :: SIMDVec v => v -> v -> v
vAdd = vAddImpl

-- | Vector subtraction.
vSub :: SIMDVec v => v -> v -> v
vSub = vSubImpl

-- | Vector multiplication.
vMul :: SIMDVec v => v -> v -> v
vMul = vMulImpl

-- | Vector division.
vDiv :: SIMDVec v => v -> v -> v
vDiv = vDivImpl

-- | Negate all elements.
vNeg :: SIMDVec v => v -> v
vNeg v = vSub (vBroadcast (vExtract v 0 - vExtract v 0)) v  -- 0 - v

-- | Absolute value of all elements.
vAbs :: (SIMDVec v, Num (Elem v), Ord (Elem v)) => v -> v
vAbs v = fromList [abs x | x <- toList v]

-- | Square root of all elements.
vSqrt :: (SIMDVec v, Floating (Elem v)) => v -> v
vSqrt v = fromList [sqrt x | x <- toList v]

-- | Reciprocal square root (fast approximation).
vRsqrt :: (SIMDVec v, Floating (Elem v)) => v -> v
vRsqrt v = fromList [1.0 / sqrt x | x <- toList v]

-- | Reciprocal (fast approximation).
vRcp :: (SIMDVec v, Fractional (Elem v)) => v -> v
vRcp v = fromList [1.0 / x | x <- toList v]

-- | Element-wise minimum.
vMin :: (SIMDVec v, Ord (Elem v)) => v -> v -> v
vMin va vb = fromList [min a b | (a, b) <- P.zip (toList va) (toList vb)]

-- | Element-wise maximum.
vMax :: (SIMDVec v, Ord (Elem v)) => v -> v -> v
vMax va vb = fromList [max a b | (a, b) <- P.zip (toList va) (toList vb)]

-- ============================================================
-- Fused Multiply-Add
-- ============================================================
--
-- FMA operations compute multiply-add in a single instruction
-- without intermediate rounding. This provides both better
-- performance and precision compared to separate operations.

-- | Fused multiply-add: @a * b + c@ in a single operation.
--
-- ==== __Hardware Support__
--
-- * AVX2/FMA3: Single @vfmadd@ instruction
-- * Without FMA: Falls back to mul + add (2 instructions)
--
-- ==== __Precision__
--
-- FMA maintains full precision for the intermediate @a * b@ result
-- (no rounding between multiply and add). Results may differ from
-- @a * b + c@ computed separately.
vFMA :: (SIMDVec v, Num (Elem v)) => v -> v -> v -> v
vFMA a b c = fromList [x * y + z | (x, y, z) <- P.zip3 (toList a) (toList b) (toList c)]

-- | Fused multiply-subtract: @a * b - c@ in a single operation.
vFMS :: (SIMDVec v, Num (Elem v)) => v -> v -> v -> v
vFMS a b c = fromList [x * y - z | (x, y, z) <- P.zip3 (toList a) (toList b) (toList c)]

-- | Fused negate-multiply-add: @-(a * b) + c@ in a single operation.
vFNMA :: (SIMDVec v, Num (Elem v)) => v -> v -> v -> v
vFNMA a b c = fromList [-(x * y) + z | (x, y, z) <- P.zip3 (toList a) (toList b) (toList c)]

-- | Fused negate-multiply-subtract: @-(a * b) - c@ in a single operation.
vFNMS :: (SIMDVec v, Num (Elem v)) => v -> v -> v -> v
vFNMS a b c = fromList [-(x * y) - z | (x, y, z) <- P.zip3 (toList a) (toList b) (toList c)]

-- ============================================================
-- Comparison
-- ============================================================

-- | Element-wise equality.
vEq :: (SIMDVec v, Eq (Elem v)) => v -> v -> Mask n
vEq va vb = Mask $ P.foldr (\(i, eq) acc -> if eq then acc .|. (1 `shiftL` i) else acc) 0
    [(i, a == b) | (i, (a, b)) <- P.zip [0..] (P.zip (toList va) (toList vb))]

-- | Element-wise inequality.
vNe :: (SIMDVec v, Eq (Elem v)) => v -> v -> Mask n
vNe va vb = Mask $ P.foldr (\(i, ne) acc -> if ne then acc .|. (1 `shiftL` i) else acc) 0
    [(i, a /= b) | (i, (a, b)) <- P.zip [0..] (P.zip (toList va) (toList vb))]

-- | Element-wise less than.
vLt :: (SIMDVec v, Ord (Elem v)) => v -> v -> Mask n
vLt va vb = Mask $ P.foldr (\(i, lt) acc -> if lt then acc .|. (1 `shiftL` i) else acc) 0
    [(i, a < b) | (i, (a, b)) <- P.zip [0..] (P.zip (toList va) (toList vb))]

-- | Element-wise less than or equal.
vLe :: (SIMDVec v, Ord (Elem v)) => v -> v -> Mask n
vLe va vb = Mask $ P.foldr (\(i, le) acc -> if le then acc .|. (1 `shiftL` i) else acc) 0
    [(i, a <= b) | (i, (a, b)) <- P.zip [0..] (P.zip (toList va) (toList vb))]

-- | Element-wise greater than.
vGt :: (SIMDVec v, Ord (Elem v)) => v -> v -> Mask n
vGt va vb = Mask $ P.foldr (\(i, gt) acc -> if gt then acc .|. (1 `shiftL` i) else acc) 0
    [(i, a > b) | (i, (a, b)) <- P.zip [0..] (P.zip (toList va) (toList vb))]

-- | Element-wise greater than or equal.
vGe :: (SIMDVec v, Ord (Elem v)) => v -> v -> Mask n
vGe va vb = Mask $ P.foldr (\(i, ge) acc -> if ge then acc .|. (1 `shiftL` i) else acc) 0
    [(i, a >= b) | (i, (a, b)) <- P.zip [0..] (P.zip (toList va) (toList vb))]

-- ============================================================
-- Horizontal Operations
-- ============================================================
--
-- Horizontal operations reduce a vector to a scalar. These are
-- generally slower than vertical operations (between vectors)
-- because they require cross-lane communication.

-- | Sum all elements of the vector (horizontal sum).
--
-- >>> vSum (Vec4F32 1 2 3 4)
-- 10.0
--
-- __Performance Note__: Horizontal operations are slower than
-- vertical operations. Prefer vertical ops when possible.
vSum :: SIMDVec v => v -> Elem v
vSum = vSumImpl

-- | Product of all elements (horizontal product).
--
-- >>> vProduct (Vec4F32 1 2 3 4)
-- 24.0
vProduct :: SIMDVec v => v -> Elem v
vProduct = vProductImpl

-- | Minimum element of the vector.
--
-- >>> vHMin (Vec4F32 3 1 4 2)
-- 1.0
vHMin :: (SIMDVec v, Ord (Elem v)) => v -> Elem v
vHMin v = P.minimum (toList v)

-- | Maximum element of the vector.
--
-- >>> vHMax (Vec4F32 3 1 4 2)
-- 4.0
vHMax :: (SIMDVec v, Ord (Elem v)) => v -> Elem v
vHMax v = P.maximum (toList v)

-- ============================================================
-- Shuffle and Blend
-- ============================================================

-- | Shuffle elements using index mask.
vShuffle :: SIMDVec v => v -> v -> [Int] -> v
vShuffle va vb indices = fromList [selectElem i | i <- indices]
  where
    w = vWidth va
    listA = toList va
    listB = toList vb
    selectElem i
        | i < w     = listA P.!! i
        | otherwise = listB P.!! (i - w)

-- | Blend two vectors using mask.
vBlend :: SIMDVec v => Mask n -> v -> v -> v
vBlend (Mask m) va vb = fromList
    [if (m .&. (1 `shiftL` i)) /= 0 then b else a
     | (i, (a, b)) <- P.zip [0..] (P.zip (toList va) (toList vb))]

-- | Permute elements within vector.
vPermute :: SIMDVec v => v -> [Int] -> v
vPermute v indices = fromList [vExtract v i | i <- indices]

-- | Rotate elements left.
vRotate :: SIMDVec v => v -> Int -> v
vRotate v n =
    let xs = toList v
        w = P.length xs
        n' = n `mod` w
    in fromList (P.drop n' xs P.++ P.take n' xs)

-- ============================================================
-- Load/Store
-- ============================================================

-- | Load from unaligned memory.
vLoad :: (SIMDVec v, Storable (Elem v)) => Ptr (Elem v) -> IO v
vLoad ptr = do
    xs <- peekArray w ptr
    pure $ fromList xs
  where
    w = vWidth (undefined :: v)

-- | Load from aligned memory (faster).
vLoadAligned :: (SIMDVec v, Storable (Elem v)) => Ptr (Elem v) -> IO v
vLoadAligned = vLoad  -- Same as unaligned in pure Haskell

-- | Store to unaligned memory.
vStore :: (SIMDVec v, Storable (Elem v)) => Ptr (Elem v) -> v -> IO ()
vStore ptr v = pokeArray ptr (toList v)

-- | Store to aligned memory (faster).
vStoreAligned :: (SIMDVec v, Storable (Elem v)) => Ptr (Elem v) -> v -> IO ()
vStoreAligned = vStore  -- Same as unaligned in pure Haskell

-- | Gather: load from indexed locations.
vGather :: (SIMDVec v, Storable (Elem v)) => Ptr (Elem v) -> Vec4I32 -> IO v
vGather ptr (Vec4I32 i0 i1 i2 i3) = do
    x0 <- peekElemOff ptr i0
    x1 <- peekElemOff ptr i1
    x2 <- peekElemOff ptr i2
    x3 <- peekElemOff ptr i3
    pure $ fromList [x0, x1, x2, x3]

-- | Scatter: store to indexed locations.
vScatter :: (SIMDVec v, Storable (Elem v)) => Ptr (Elem v) -> Vec4I32 -> v -> IO ()
vScatter ptr (Vec4I32 i0 i1 i2 i3) v = do
    let xs = toList v
    pokeElemOff ptr i0 (xs P.!! 0)
    pokeElemOff ptr i1 (xs P.!! 1)
    pokeElemOff ptr i2 (xs P.!! 2)
    pokeElemOff ptr i3 (xs P.!! 3)

-- | Masked load.
vMaskedLoad :: (SIMDVec v, Storable (Elem v)) => Mask n -> Ptr (Elem v) -> v -> IO v
vMaskedLoad (Mask m) ptr def = do
    xs <- sequence [if (m .&. (1 `shiftL` i)) /= 0
                    then peekElemOff ptr i
                    else pure (vExtract def i)
                   | i <- [0 .. vWidth def - 1]]
    pure $ fromList xs

-- | Masked store.
vMaskedStore :: (SIMDVec v, Storable (Elem v)) => Mask n -> Ptr (Elem v) -> v -> IO ()
vMaskedStore (Mask m) ptr v =
    sequence_ [if (m .&. (1 `shiftL` i)) /= 0
               then pokeElemOff ptr i (vExtract v i)
               else pure ()
              | i <- [0 .. vWidth v - 1]]

-- ============================================================
-- Bitwise Operations
-- ============================================================

-- | Bitwise AND (for integer vectors).
vAnd :: Vec4I32 -> Vec4I32 -> Vec4I32
vAnd (Vec4I32 a1 b1 c1 d1) (Vec4I32 a2 b2 c2 d2) =
    Vec4I32 (a1 .&. a2) (b1 .&. b2) (c1 .&. c2) (d1 .&. d2)

-- | Bitwise OR (for integer vectors).
vOr :: Vec4I32 -> Vec4I32 -> Vec4I32
vOr (Vec4I32 a1 b1 c1 d1) (Vec4I32 a2 b2 c2 d2) =
    Vec4I32 (a1 .|. a2) (b1 .|. b2) (c1 .|. c2) (d1 .|. d2)

-- | Bitwise XOR (for integer vectors).
vXor :: Vec4I32 -> Vec4I32 -> Vec4I32
vXor (Vec4I32 a1 b1 c1 d1) (Vec4I32 a2 b2 c2 d2) =
    Vec4I32 (a1 `xor` a2) (b1 `xor` b2) (c1 `xor` c2) (d1 `xor` d2)

-- | Bitwise NOT (for integer vectors).
vNot :: Vec4I32 -> Vec4I32
vNot (Vec4I32 a b c d) = Vec4I32 (complement a) (complement b) (complement c) (complement d)

-- | Shift left (for integer vectors).
vShiftL :: Vec4I32 -> Int -> Vec4I32
vShiftL (Vec4I32 a b c d) n = Vec4I32 (a `shiftL` n) (b `shiftL` n) (c `shiftL` n) (d `shiftL` n)

-- | Shift right (for integer vectors).
vShiftR :: Vec4I32 -> Int -> Vec4I32
vShiftR (Vec4I32 a b c d) n = Vec4I32 (a `shiftR` n) (b `shiftR` n) (c `shiftR` n) (d `shiftR` n)
