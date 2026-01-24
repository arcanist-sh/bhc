-- |
-- Module      : BHC.Data.Random
-- Description : Random number generation
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Fast, high-quality random number generation.
--
-- = Overview
--
-- This module provides a fast PRNG (pseudo-random number generator)
-- suitable for simulations, games, and general-purpose randomness.
-- For cryptographic purposes, use a dedicated crypto library.
--
-- = Quick Start
--
-- @
-- import BHC.Data.Random
--
-- main :: IO ()
-- main = do
--     rng <- newRng           -- Seed from system entropy
--     n <- range rng 1 100    -- Random int 1-100
--     f <- nextFloat rng      -- Random float [0, 1)
--     b <- nextBool rng       -- Random boolean
--     print (n, f, b)
-- @
--
-- = Reproducibility
--
-- For reproducible results (testing, simulations), use 'fromSeed':
--
-- @
-- rng <- fromSeed 12345
-- -- Same sequence every time with this seed
-- @
--
-- = Distributions
--
-- * 'uniform' — Uniform distribution in [0, 1)
-- * 'normal' — Normal (Gaussian) distribution
-- * 'exponential' — Exponential distribution
--
-- = Collection Operations
--
-- * 'choose' — Pick a random element
-- * 'shuffle' — Randomly reorder a list
-- * 'sample' — Pick n elements without replacement

module BHC.Data.Random (
    -- * Random number generator
    Rng,
    newRng,
    fromSeed,
    
    -- * Generation
    nextInt,
    nextFloat,
    nextDouble,
    nextBool,
    
    -- * Ranges
    range,
    rangeFloat,
    rangeDouble,
    
    -- * Collections
    choose,
    shuffle,
    sample,
    
    -- * Distributions
    uniform,
    normal,
    exponential,
    
    -- * Utilities
    uuid,
) where

import BHC.Prelude

-- | A random number generator.
data Rng = Rng
    { rngState :: !(IORef (Word64, Word64))
    }

-- | Create a new Rng seeded from system entropy.
foreign import ccall "bhc_rng_new" newRng :: IO Rng

-- | Create an Rng from a seed.
foreign import ccall "bhc_rng_from_seed" fromSeed :: Word64 -> IO Rng

-- | Generate a random Int.
foreign import ccall "bhc_rng_next_int" nextInt :: Rng -> IO Int

-- | Generate a random Float in [0, 1).
foreign import ccall "bhc_rng_next_float" nextFloat :: Rng -> IO Float

-- | Generate a random Double in [0, 1).
foreign import ccall "bhc_rng_next_double" nextDouble :: Rng -> IO Double

-- | Generate a random Bool.
nextBool :: Rng -> IO Bool
nextBool rng = do
    n <- nextInt rng
    return (n `mod` 2 == 0)

-- | Generate a random Int in [min, max].
range :: Rng -> Int -> Int -> IO Int
range rng minV maxV = do
    n <- nextInt rng
    return $ minV + (abs n `mod` (maxV - minV + 1))

-- | Generate a random Float in [min, max).
rangeFloat :: Rng -> Float -> Float -> IO Float
rangeFloat rng minV maxV = do
    f <- nextFloat rng
    return $ minV + f * (maxV - minV)

-- | Generate a random Double in [min, max).
rangeDouble :: Rng -> Double -> Double -> IO Double
rangeDouble rng minV maxV = do
    d <- nextDouble rng
    return $ minV + d * (maxV - minV)

-- | Choose a random element from a list.
choose :: Rng -> [a] -> IO (Maybe a)
choose _ []  = return Nothing
choose rng xs = do
    i <- range rng 0 (length xs - 1)
    return (Just (xs !! i))

-- | Shuffle a list.
shuffle :: Rng -> [a] -> IO [a]
shuffle _ []  = return []
shuffle rng xs = go (length xs) xs
  where
    go 0 acc = return acc
    go n acc = do
        i <- range rng 0 (n - 1)
        let (before, x:after) = splitAt i acc
        rest <- go (n - 1) (before ++ after)
        return (x : rest)

-- | Sample n elements from a list (without replacement).
sample :: Rng -> Int -> [a] -> IO [a]
sample rng n xs = take n <$> shuffle rng xs

-- | Uniform distribution in [0, 1).
uniform :: Rng -> IO Double
uniform = nextDouble

-- | Normal distribution with given mean and standard deviation.
-- Uses Box-Muller transform.
normal :: Rng -> Double -> Double -> IO Double
normal rng mean stddev = do
    u1 <- nextDouble rng
    u2 <- nextDouble rng
    let z = sqrt (-2 * log u1) * cos (2 * pi * u2)
    return (mean + stddev * z)

-- | Exponential distribution with given rate.
exponential :: Rng -> Double -> IO Double
exponential rng rate = do
    u <- nextDouble rng
    return (- log u / rate)

-- | Generate a UUID v4.
uuid :: Rng -> IO String
uuid rng = do
    bytes <- replicateM 16 (nextInt rng)
    let hex = concatMap (pad2Hex . (`mod` 256)) bytes
        (a, rest1) = splitAt 8 hex
        (b, rest2) = splitAt 4 rest1
        (c, rest3) = splitAt 4 rest2
        (d, e) = splitAt 4 rest3
    return $ a ++ "-" ++ b ++ "-" ++ c ++ "-" ++ d ++ "-" ++ take 12 e

pad2Hex :: Int -> String
pad2Hex n = let h = hexDigits n in if length h == 1 then '0' : h else h

hexDigits :: Int -> String
hexDigits n
    | n < 16 = [hexDigit n]
    | otherwise = hexDigits (n `div` 16) ++ [hexDigit (n `mod` 16)]

hexDigit :: Int -> Char
hexDigit n
    | n < 10 = toEnum (fromEnum '0' + n)
    | otherwise = toEnum (fromEnum 'a' + n - 10)

-- Internal
data IORef a
type Word64 = Int  -- placeholder
