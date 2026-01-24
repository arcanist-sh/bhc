-- |
-- Module      : H26.Numeric
-- Description : Numeric types and operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Numeric types optimized for high-performance computing.
--
-- = Overview
--
-- This module provides comprehensive numeric support including:
--
-- * Primitive types: Int8-64, Word8-64, Float, Double, Half
-- * Fixed-point arithmetic for DSP applications
-- * Complex numbers
-- * SIMD vector types (SSE, AVX, AVX-512)
-- * Accurate summation algorithms
-- * Statistical functions
--
-- = Quick Start
--
-- @
-- {-# PROFILE Numeric #-}
-- import H26.Numeric
--
-- -- SIMD operations (auto-vectorized)
-- dotProduct :: [Float] -> [Float] -> Float
-- dotProduct xs ys = kahanSum (zipWith (*) xs ys)
--
-- -- Complex numbers
-- z :: Complex Double
-- z = 3 :+ 4              -- 3 + 4i
-- r = magnitude z         -- 5.0
-- θ = phase z             -- atan(4/3)
--
-- -- SIMD vectors (explicit)
-- v1 = vec4f32 1 2 3 4
-- v2 = vec4f32 5 6 7 8
-- v3 = vMul v1 v2         -- [5, 12, 21, 32]
-- s = vSum v3             -- 70
-- @
--
-- = SIMD Vector Types
--
-- | Type      | Width   | Elements | Instructions |
-- |-----------|---------|----------|--------------|
-- | Vec4F32   | 128-bit | 4×f32    | SSE          |
-- | Vec8F32   | 256-bit | 8×f32    | AVX          |
-- | Vec16F32  | 512-bit | 16×f32   | AVX-512      |
-- | Vec2F64   | 128-bit | 2×f64    | SSE          |
-- | Vec4F64   | 256-bit | 4×f64    | AVX          |
-- | Vec8F64   | 512-bit | 8×f64    | AVX-512      |
--
-- = Accurate Summation
--
-- For floating-point summation, use compensated algorithms:
--
-- @
-- -- Naive sum accumulates error
-- naiveSum [1e16, 1, -1e16]  -- May give 0 due to precision loss
--
-- -- Kahan summation compensates for error
-- kahanSum [1e16, 1, -1e16]  -- Gives 1
-- @
--
-- = See Also
--
-- * "H26.Tensor" for tensor operations
-- * "H26.BLAS" for linear algebra
-- * "BHC.Numeric.SIMD" for SIMD implementation

{-# HASKELL_EDITION 2026 #-}
{-# PROFILE Numeric #-}

module H26.Numeric
  ( -- * Primitive Numeric Types
    Int8, Int16, Int32, Int64
  , Word8, Word16, Word32, Word64
  , Float, Double
  , Half  -- 16-bit float

    -- * Natural and Integer
  , Natural
  , Integer

    -- * Fixed-Point Numbers
  , Fixed
  , FixedS16  -- 16-bit signed fixed
  , FixedS32  -- 32-bit signed fixed
  , FixedU16  -- 16-bit unsigned fixed
  , FixedU32  -- 32-bit unsigned fixed
  , resolution
  , mkFixed
  , fromFixed
  , fixedToRational

    -- * Complex Numbers
  , Complex(..)
  , realPart
  , imagPart
  , mkPolar
  , cis
  , polar
  , magnitude
  , phase
  , conjugate

    -- * Rational Numbers
  , Ratio
  , Rational
  , (%)
  , numerator
  , denominator
  , approxRational

    -- * SIMD Vector Types
  , Vec2F32, Vec4F32, Vec8F32, Vec16F32
  , Vec2F64, Vec4F64, Vec8F64
  , Vec2I32, Vec4I32, Vec8I32, Vec16I32
  , Vec2I64, Vec4I64, Vec8I64

    -- * SIMD Construction
  , vec2f32, vec4f32, vec8f32, vec16f32
  , vec2f64, vec4f64, vec8f64
  , vec2i32, vec4i32, vec8i32, vec16i32
  , vec2i64, vec4i64, vec8i64
  , broadcast
  , fromList
  , toList

    -- * SIMD Element Access
  , extract
  , insert
  , shuffle
  , permute

    -- * SIMD Arithmetic
  , vAdd, vSub, vMul, vDiv
  , vNeg, vAbs, vMin, vMax
  , vSqrt, vRsqrt, vRcp
  , vFMA, vFMS  -- Fused multiply-add/sub

    -- * SIMD Comparison
  , vEq, vNe, vLt, vLe, vGt, vGe
  , vAnd, vOr, vXor, vAndNot
  , vSelect  -- Blend based on mask

    -- * SIMD Reductions
  , vSum, vProduct
  , vMin1, vMax1
  , vHAdd, vHSub  -- Horizontal add/sub

    -- * SIMD Load/Store
  , vLoad, vLoadAligned
  , vStore, vStoreAligned
  , vMaskedLoad, vMaskedStore
  , vGather, vScatter

    -- * Numeric Classes
  , Num(..)
  , Real(..)
  , Integral(..)
  , Fractional(..)
  , Floating(..)
  , RealFrac(..)
  , RealFloat(..)

    -- * Additional Classes
  , NumericPrimitive
  , Unboxed
  , SimdPrimitive

    -- * Conversions
  , fromIntegral
  , realToFrac
  , truncate, round, ceiling, floor
  , toInteger, fromInteger
  , toRational, fromRational

    -- * Bit Operations
  , Bits(..)
  , FiniteBits(..)
  , (.&.), (.|.), xor
  , complement
  , shift, shiftL, shiftR
  , rotate, rotateL, rotateR
  , bit, setBit, clearBit, complementBit
  , testBit, bitSize, bitSizeMaybe
  , isSigned, popCount
  , countLeadingZeros, countTrailingZeros

    -- * Floating Point Utilities
  , isNaN, isInfinite, isDenormalized, isNegativeZero
  , isIEEE
  , floatRadix, floatDigits, floatRange
  , decodeFloat, encodeFloat
  , exponent, significand, scaleFloat
  , atan2

    -- * Special Values
  , infinity, negInfinity, nan
  , minNormal, maxFinite
  , epsilon, pi, e

    -- * Numeric Utilities
  , gcd, lcm
  , abs, signum, negate
  , subtract
  , even, odd
  , (^), (^^)

    -- * Summation Algorithms
  , sum, product
  , kahanSum       -- Compensated summation
  , pairwiseSum    -- Binary tree summation
  , neumaierSum    -- Improved Kahan

    -- * Statistical Functions
  , mean
  , variance
  , stddev
  , covariance
  , correlation

    -- * Range Generation
  , enumFromTo
  , enumFromThenTo
  , range
  , linspace
  , logspace
  , geomspace

    -- * Constants
  , maxBound, minBound

    -- * Rounding Modes
  , RoundingMode(..)
  , setRoundingMode
  , getRoundingMode
  , withRoundingMode

    -- * Floating Point Exceptions
  , FPException(..)
  , getFPExceptions
  , clearFPExceptions
  , enableFPException
  , disableFPException
  ) where

import Prelude hiding
  ( Num(..), Real(..), Integral(..), Fractional(..), Floating(..)
  , RealFrac(..), RealFloat(..), fromIntegral, realToFrac
  , truncate, round, ceiling, floor, toInteger, fromInteger
  , toRational, fromRational, gcd, lcm, abs, signum, negate
  , subtract, even, odd, (^), sum, product, maxBound, minBound
  , enumFromTo, enumFromThenTo
  )

-- | 16-bit floating point (IEEE 754 half precision).
data Half

-- | Fixed-point number with resolution r.
newtype Fixed r = Fixed Integer

-- | 16-bit signed fixed point (1/65536 resolution).
type FixedS16 = Fixed S16

-- | 32-bit signed fixed point (1/4294967296 resolution).
type FixedS32 = Fixed S32

-- | 16-bit unsigned fixed point.
type FixedU16 = Fixed U16

-- | 32-bit unsigned fixed point.
type FixedU32 = Fixed U32

-- | Get resolution of fixed-point type.
resolution :: Fixed r -> Integer

-- | Create fixed-point from integer.
mkFixed :: Integer -> Fixed r

-- | Convert fixed to integer (truncates).
fromFixed :: Fixed r -> Integer

-- | Convert fixed to exact Rational.
fixedToRational :: Fixed r -> Rational

-- | Complex number.
data Complex a = !a :+ !a
  deriving (Eq, Show, Read)

-- | Extract real part.
realPart :: Complex a -> a

-- | Extract imaginary part.
imagPart :: Complex a -> a

-- | Create from polar coordinates.
mkPolar :: Floating a => a -> a -> Complex a

-- | Unit complex from angle.
cis :: Floating a => a -> Complex a

-- | Convert to polar form (magnitude, phase).
polar :: RealFloat a => Complex a -> (a, a)

-- | Magnitude (absolute value).
magnitude :: RealFloat a => Complex a -> a

-- | Phase angle.
phase :: RealFloat a => Complex a -> a

-- | Complex conjugate.
conjugate :: Num a => Complex a -> Complex a

-- | Ratio of two values.
data Ratio a = !a :% !a

-- | Rational number.
type Rational = Ratio Integer

-- | Construct ratio.
(%) :: Integral a => a -> a -> Ratio a

-- | Extract numerator.
numerator :: Ratio a -> a

-- | Extract denominator.
denominator :: Ratio a -> a

-- | Approximate rational.
approxRational :: RealFrac a => a -> a -> Rational

-- SIMD Vector Types

-- | 2-element Float vector.
data Vec2F32

-- | 4-element Float vector (SSE).
data Vec4F32

-- | 8-element Float vector (AVX).
data Vec8F32

-- | 16-element Float vector (AVX-512).
data Vec16F32

-- | 2-element Double vector.
data Vec2F64

-- | 4-element Double vector (AVX).
data Vec4F64

-- | 8-element Double vector (AVX-512).
data Vec8F64

-- | 2-element Int32 vector.
data Vec2I32

-- | 4-element Int32 vector (SSE).
data Vec4I32

-- | 8-element Int32 vector (AVX).
data Vec8I32

-- | 16-element Int32 vector (AVX-512).
data Vec16I32

-- | 2-element Int64 vector.
data Vec2I64

-- | 4-element Int64 vector (AVX).
data Vec4I64

-- | 8-element Int64 vector (AVX-512).
data Vec8I64

-- SIMD Construction

-- | Create 2-element Float vector.
vec2f32 :: Float -> Float -> Vec2F32

-- | Create 4-element Float vector.
vec4f32 :: Float -> Float -> Float -> Float -> Vec4F32

-- | Create 8-element Float vector.
vec8f32 :: Float -> Float -> Float -> Float
        -> Float -> Float -> Float -> Float -> Vec8F32

-- | Create 16-element Float vector.
vec16f32 :: Float -> Float -> Float -> Float
         -> Float -> Float -> Float -> Float
         -> Float -> Float -> Float -> Float
         -> Float -> Float -> Float -> Float -> Vec16F32

-- | Create 2-element Double vector.
vec2f64 :: Double -> Double -> Vec2F64

-- | Create 4-element Double vector.
vec4f64 :: Double -> Double -> Double -> Double -> Vec4F64

-- | Create 8-element Double vector.
vec8f64 :: Double -> Double -> Double -> Double
        -> Double -> Double -> Double -> Double -> Vec8F64

-- | Create 2-element Int32 vector.
vec2i32 :: Int32 -> Int32 -> Vec2I32

-- | Create 4-element Int32 vector.
vec4i32 :: Int32 -> Int32 -> Int32 -> Int32 -> Vec4I32

-- | Create 8-element Int32 vector.
vec8i32 :: Int32 -> Int32 -> Int32 -> Int32
        -> Int32 -> Int32 -> Int32 -> Int32 -> Vec8I32

-- | Create 16-element Int32 vector.
vec16i32 :: Int32 -> Int32 -> Int32 -> Int32
         -> Int32 -> Int32 -> Int32 -> Int32
         -> Int32 -> Int32 -> Int32 -> Int32
         -> Int32 -> Int32 -> Int32 -> Int32 -> Vec16I32

-- | Create 2-element Int64 vector.
vec2i64 :: Int64 -> Int64 -> Vec2I64

-- | Create 4-element Int64 vector.
vec4i64 :: Int64 -> Int64 -> Int64 -> Int64 -> Vec4I64

-- | Create 8-element Int64 vector.
vec8i64 :: Int64 -> Int64 -> Int64 -> Int64
        -> Int64 -> Int64 -> Int64 -> Int64 -> Vec8I64

-- | Broadcast scalar to all elements.
broadcast :: SimdPrimitive v a => a -> v

-- | Create vector from list.
fromList :: SimdPrimitive v a => [a] -> v

-- | Convert vector to list.
toList :: SimdPrimitive v a => v -> [a]

-- SIMD Element Operations

-- | Extract element at index.
extract :: SimdPrimitive v a => Int -> v -> a

-- | Insert element at index.
insert :: SimdPrimitive v a => Int -> a -> v -> v

-- | Shuffle elements using mask.
shuffle :: SimdPrimitive v a => [Int] -> v -> v

-- | Permute elements.
permute :: SimdPrimitive v a => [Int] -> v -> v -> v

-- SIMD Arithmetic

-- | Vector addition.
vAdd :: SimdPrimitive v a => v -> v -> v

-- | Vector subtraction.
vSub :: SimdPrimitive v a => v -> v -> v

-- | Vector multiplication.
vMul :: SimdPrimitive v a => v -> v -> v

-- | Vector division.
vDiv :: SimdPrimitive v a => v -> v -> v

-- | Vector negation.
vNeg :: SimdPrimitive v a => v -> v

-- | Vector absolute value.
vAbs :: SimdPrimitive v a => v -> v

-- | Element-wise minimum.
vMin :: SimdPrimitive v a => v -> v -> v

-- | Element-wise maximum.
vMax :: SimdPrimitive v a => v -> v -> v

-- | Vector square root.
vSqrt :: SimdPrimitive v a => v -> v

-- | Reciprocal square root (approximate).
vRsqrt :: SimdPrimitive v a => v -> v

-- | Reciprocal (approximate).
vRcp :: SimdPrimitive v a => v -> v

-- | Fused multiply-add: a * b + c.
vFMA :: SimdPrimitive v a => v -> v -> v -> v

-- | Fused multiply-subtract: a * b - c.
vFMS :: SimdPrimitive v a => v -> v -> v -> v

-- SIMD Comparison

-- | Vector equality comparison.
vEq :: SimdPrimitive v a => v -> v -> v

-- | Vector inequality comparison.
vNe :: SimdPrimitive v a => v -> v -> v

-- | Vector less-than comparison.
vLt :: SimdPrimitive v a => v -> v -> v

-- | Vector less-equal comparison.
vLe :: SimdPrimitive v a => v -> v -> v

-- | Vector greater-than comparison.
vGt :: SimdPrimitive v a => v -> v -> v

-- | Vector greater-equal comparison.
vGe :: SimdPrimitive v a => v -> v -> v

-- | Bitwise AND.
vAnd :: SimdPrimitive v a => v -> v -> v

-- | Bitwise OR.
vOr :: SimdPrimitive v a => v -> v -> v

-- | Bitwise XOR.
vXor :: SimdPrimitive v a => v -> v -> v

-- | Bitwise AND-NOT.
vAndNot :: SimdPrimitive v a => v -> v -> v

-- | Select elements based on mask.
vSelect :: SimdPrimitive v a => v -> v -> v -> v

-- SIMD Reductions

-- | Sum all elements.
vSum :: SimdPrimitive v a => v -> a

-- | Product of all elements.
vProduct :: SimdPrimitive v a => v -> a

-- | Minimum element.
vMin1 :: SimdPrimitive v a => v -> a

-- | Maximum element.
vMax1 :: SimdPrimitive v a => v -> a

-- | Horizontal add (adjacent pairs).
vHAdd :: SimdPrimitive v a => v -> v -> v

-- | Horizontal subtract (adjacent pairs).
vHSub :: SimdPrimitive v a => v -> v -> v

-- SIMD Memory Operations

-- | Load vector from memory.
vLoad :: SimdPrimitive v a => Ptr a -> IO v

-- | Load from aligned memory.
vLoadAligned :: SimdPrimitive v a => Ptr a -> IO v

-- | Store vector to memory.
vStore :: SimdPrimitive v a => Ptr a -> v -> IO ()

-- | Store to aligned memory.
vStoreAligned :: SimdPrimitive v a => Ptr a -> v -> IO ()

-- | Masked load.
vMaskedLoad :: SimdPrimitive v a => v -> Ptr a -> v -> IO v

-- | Masked store.
vMaskedStore :: SimdPrimitive v a => Ptr a -> v -> v -> IO ()

-- | Gather from non-contiguous memory.
vGather :: SimdPrimitive v a => Ptr a -> Vec4I32 -> v

-- | Scatter to non-contiguous memory.
vScatter :: SimdPrimitive v a => Ptr a -> Vec4I32 -> v -> IO ()

-- Numeric Classes

-- | Basic numeric operations.
class Num a where
  (+), (-), (*) :: a -> a -> a
  negate :: a -> a
  abs :: a -> a
  signum :: a -> a
  fromInteger :: Integer -> a

-- | Real numbers (can convert to Rational).
class Num a => Real a where
  toRational :: a -> Rational

-- | Integral numbers.
class (Real a, Enum a) => Integral a where
  quot, rem, div, mod :: a -> a -> a
  quotRem, divMod :: a -> a -> (a, a)
  toInteger :: a -> Integer

-- | Fractional numbers.
class Num a => Fractional a where
  (/) :: a -> a -> a
  recip :: a -> a
  fromRational :: Rational -> a

-- | Floating point numbers.
class Fractional a => Floating a where
  pi :: a
  exp, log, sqrt :: a -> a
  (**), logBase :: a -> a -> a
  sin, cos, tan :: a -> a
  asin, acos, atan :: a -> a
  sinh, cosh, tanh :: a -> a
  asinh, acosh, atanh :: a -> a

-- | Real fractional numbers.
class (Real a, Fractional a) => RealFrac a where
  properFraction :: Integral b => a -> (b, a)
  truncate, round, ceiling, floor :: Integral b => a -> b

-- | Real floating point numbers.
class (RealFrac a, Floating a) => RealFloat a where
  floatRadix :: a -> Integer
  floatDigits :: a -> Int
  floatRange :: a -> (Int, Int)
  decodeFloat :: a -> (Integer, Int)
  encodeFloat :: Integer -> Int -> a
  exponent :: a -> Int
  significand :: a -> a
  scaleFloat :: Int -> a -> a
  isNaN, isInfinite, isDenormalized, isNegativeZero, isIEEE :: a -> Bool
  atan2 :: a -> a -> a

-- | Primitive numeric types.
class NumericPrimitive a

-- | Unboxed types.
class Unboxed a

-- | SIMD primitive types.
class SimdPrimitive v a | v -> a where
  simdWidth :: v -> Int

-- Conversions

-- | Convert between integral types.
fromIntegral :: (Integral a, Num b) => a -> b

-- | Convert between fractional types.
realToFrac :: (Real a, Fractional b) => a -> b

-- Bit Operations

-- | Class for types with bit operations.
class Eq a => Bits a where
  (.&.) :: a -> a -> a
  (.|.) :: a -> a -> a
  xor :: a -> a -> a
  complement :: a -> a
  shift :: a -> Int -> a
  shiftL :: a -> Int -> a
  shiftR :: a -> Int -> a
  rotate :: a -> Int -> a
  rotateL :: a -> Int -> a
  rotateR :: a -> Int -> a
  bit :: Int -> a
  setBit :: a -> Int -> a
  clearBit :: a -> Int -> a
  complementBit :: a -> Int -> a
  testBit :: a -> Int -> Bool
  bitSize :: a -> Int
  bitSizeMaybe :: a -> Maybe Int
  isSigned :: a -> Bool
  popCount :: a -> Int

-- | Bits with finite size.
class Bits a => FiniteBits a where
  finiteBitSize :: a -> Int
  countLeadingZeros :: a -> Int
  countTrailingZeros :: a -> Int

-- Special Values

-- | Positive infinity.
infinity :: RealFloat a => a

-- | Negative infinity.
negInfinity :: RealFloat a => a

-- | Not a number.
nan :: RealFloat a => a

-- | Smallest positive normal.
minNormal :: RealFloat a => a

-- | Largest finite value.
maxFinite :: RealFloat a => a

-- | Machine epsilon.
epsilon :: RealFloat a => a

-- | Euler's number.
e :: Floating a => a

-- Numeric Utilities

-- | Greatest common divisor.
gcd :: Integral a => a -> a -> a

-- | Least common multiple.
lcm :: Integral a => a -> a -> a

-- | Subtraction (flip of -).
subtract :: Num a => a -> a -> a

-- | Test if even.
even :: Integral a => a -> Bool

-- | Test if odd.
odd :: Integral a => a -> Bool

-- | Integer power.
(^) :: (Num a, Integral b) => a -> b -> a

-- | Fractional power.
(^^) :: (Fractional a, Integral b) => a -> b -> a

-- Summation

-- | Sum of list.
sum :: Num a => [a] -> a

-- | Product of list.
product :: Num a => [a] -> a

-- | Kahan compensated summation.
--
-- More accurate than naive sum for floating point.
kahanSum :: RealFloat a => [a] -> a

-- | Pairwise summation.
--
-- O(n log n) accuracy for floating point.
pairwiseSum :: RealFloat a => [a] -> a

-- | Neumaier summation (improved Kahan).
neumaierSum :: RealFloat a => [a] -> a

-- Statistics

-- | Arithmetic mean.
mean :: Fractional a => [a] -> a

-- | Sample variance.
variance :: Floating a => [a] -> a

-- | Standard deviation.
stddev :: Floating a => [a] -> a

-- | Sample covariance.
covariance :: Floating a => [a] -> [a] -> a

-- | Pearson correlation coefficient.
correlation :: Floating a => [a] -> [a] -> a

-- Range Generation

-- | Enumeration from start to end.
enumFromTo :: Enum a => a -> a -> [a]

-- | Enumeration with step.
enumFromThenTo :: Enum a => a -> a -> a -> [a]

-- | Integer range.
range :: Int -> Int -> [Int]

-- | Linear spacing.
linspace :: Fractional a => a -> a -> Int -> [a]

-- | Logarithmic spacing.
logspace :: Floating a => a -> a -> Int -> [a]

-- | Geometric spacing.
geomspace :: Floating a => a -> a -> Int -> [a]

-- Bounds

-- | Maximum value for bounded type.
maxBound :: Bounded a => a

-- | Minimum value for bounded type.
minBound :: Bounded a => a

-- Rounding Modes

-- | IEEE 754 rounding modes.
data RoundingMode
  = RoundNearest      -- ^ Round to nearest, ties to even
  | RoundDown         -- ^ Round toward negative infinity
  | RoundUp           -- ^ Round toward positive infinity
  | RoundToZero       -- ^ Round toward zero
  deriving (Eq, Show, Enum, Bounded)

-- | Set current rounding mode.
setRoundingMode :: RoundingMode -> IO ()

-- | Get current rounding mode.
getRoundingMode :: IO RoundingMode

-- | Execute with specific rounding mode.
withRoundingMode :: RoundingMode -> IO a -> IO a

-- Floating Point Exceptions

-- | Floating point exception flags.
data FPException
  = FPInvalid        -- ^ Invalid operation (e.g., sqrt(-1))
  | FPDivByZero      -- ^ Division by zero
  | FPOverflow       -- ^ Result too large
  | FPUnderflow      -- ^ Result too small
  | FPInexact        -- ^ Rounded result
  deriving (Eq, Show, Enum, Bounded)

-- | Get current FP exception flags.
getFPExceptions :: IO [FPException]

-- | Clear all FP exception flags.
clearFPExceptions :: IO ()

-- | Enable trapping for exception.
enableFPException :: FPException -> IO ()

-- | Disable trapping for exception.
disableFPException :: FPException -> IO ()

-- Internal types
class Enum a
class Bounded a
data Ptr a
data S16
data S32
data U16
data U32

-- This is a specification file.
-- Actual implementation provided by the compiler.
