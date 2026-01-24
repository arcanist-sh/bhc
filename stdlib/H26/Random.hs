-- |
-- Module      : H26.Random
-- Description : Random number generation
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Random number generation with reproducibility and distributions.
--
-- = Overview
--
-- This module provides a fast PRNG (pseudo-random number generator)
-- suitable for simulations, games, and numeric workloads. Features:
--
-- * High-quality SplitMix algorithm
-- * Reproducible sequences from seeds
-- * Various distributions (uniform, normal, exponential, etc.)
-- * Efficient tensor generation (Numeric Profile)
--
-- __Note__: For cryptographic purposes, use a dedicated crypto library.
--
-- = Quick Start
--
-- @
-- import H26.Random
--
-- main :: IO ()
-- main = do
--     -- Random values using global generator
--     n <- randomRIO (1, 100)    -- Int in [1, 100]
--     f <- randomIO              -- Random Float
--
--     -- Reproducible sequence from seed
--     let g = mkStdGen 12345
--         (x, g') = random g     -- Same x every time
--
--     -- Shuffle a list
--     shuffled <- shuffleM [1..10]
--
--     print (n, f, x, shuffled)
-- @
--
-- = Distributions
--
-- @
-- -- Uniform in [0, 1)
-- u <- getStdRandom uniformDouble01M
--
-- -- Normal (Gaussian) with mean 0, stddev 1
-- n <- getStdRandom standardNormal
--
-- -- Normal with custom mean and stddev
-- n' <- getStdRandom (normalM 100 15)
--
-- -- Exponential with rate λ
-- e <- getStdRandom (exponentialM 0.5)
-- @
--
-- = Tensor Generation (Numeric Profile)
--
-- Efficiently generate random tensors:
--
-- @
-- {-# PROFILE Numeric #-}
-- import H26.Random
-- import H26.Tensor
--
-- g <- newStdGen
-- let (t1, g') = randomTensor [100, 100] g      -- Uniform [0,1)
-- let (t2, _) = normalTensor 0 1 [100, 100] g'  -- Standard normal
-- @
--
-- = See Also
--
-- * "BHC.Data.Random" for the underlying implementation
-- * "H26.Tensor" for tensor operations

{-# HASKELL_EDITION 2026 #-}

module H26.Random
  ( -- * Random Generator
    StdGen
  , RandomGen(..)

    -- * Initialization
  , newStdGen
  , mkStdGen
  , getStdGen
  , setStdGen
  , getStdRandom

    -- * Random Class
  , Random(..)
  , Uniform
  , UniformRange

    -- * Basic Generation
  , random
  , randomR
  , randoms
  , randomRs
  , randomIO
  , randomRIO

    -- * Uniform Generation
  , uniformM
  , uniformRM
  , uniformByteStringM

    -- * Distributions
  , uniformDouble01M
  , uniformDoublePositive01M
  , uniformFloat01M
  , uniformFloatPositive01M

    -- * Normal Distribution
  , normalM
  , normalPair
  , standardNormal

    -- * Other Distributions
  , exponentialM
  , gammaM
  , betaM
  , poissonM
  , bernoulliM
  , categoricalM

    -- * Shuffling
  , shuffle
  , shuffleM
  , sample
  , sampleM

    -- * Splitting
  , split
  , splitN

    -- * Tensor Random (Numeric Profile)
  , randomTensor
  , randomTensorR
  , normalTensor
  , uniformTensor

    -- * Seeds
  , Seed
  , seedFromInteger
  , seedToInteger
  , seedFromByteString
  , seedToByteString

    -- * State Monad Interface
  , Rand
  , RandT
  , runRand
  , runRandT
  , evalRand
  , evalRandT
  , execRand
  , execRandT
  , liftRand
  , liftRandT

    -- * Re-exports
  , MonadRandom(..)
  ) where

-- | Standard random number generator.
--
-- Implements a fast, high-quality PRNG (SplitMix).
-- Suitable for simulations but not for cryptographic use.
data StdGen

-- | Class of random number generators.
class RandomGen g where
  -- | Generate a random value and new generator.
  genWord64 :: g -> (Word64, g)

  -- | Generate a random value in range and new generator.
  genWord64R :: Word64 -> g -> (Word64, g)

  -- | Generate n random bytes.
  genByteString :: Int -> g -> (ByteString, g)

  -- | Split generator into two independent generators.
  split :: g -> (g, g)

-- | Class of randomly generatable types.
class Random a where
  -- | Generate a random value.
  randomM :: RandomGen g => g -> (a, g)

  -- | Generate a random value in range.
  randomRM :: RandomGen g => (a, a) -> g -> (a, g)

-- | Uniform distribution over bounded types.
class Uniform a where
  uniformM :: RandomGen g => g -> (a, g)

-- | Uniform distribution over a range.
class UniformRange a where
  uniformRM :: RandomGen g => (a, a) -> g -> (a, g)

-- | Create a new random generator from system entropy.
newStdGen :: IO StdGen

-- | Create a deterministic generator from a seed.
mkStdGen :: Int -> StdGen

-- | Get the global random generator.
getStdGen :: IO StdGen

-- | Set the global random generator.
setStdGen :: StdGen -> IO ()

-- | Apply function to global generator, returning result.
getStdRandom :: (StdGen -> (a, StdGen)) -> IO a

-- | Generate a random value using the global generator.
random :: Random a => IO a

-- | Generate a random value in range using the global generator.
randomR :: Random a => (a, a) -> IO a

-- | Generate infinite list of random values.
randoms :: (Random a, RandomGen g) => g -> [a]

-- | Generate infinite list of random values in range.
randomRs :: (Random a, RandomGen g) => (a, a) -> g -> [a]

-- | Generate random value in IO.
randomIO :: Random a => IO a

-- | Generate random value in range in IO.
randomRIO :: Random a => (a, a) -> IO a

-- | Generate a Double uniformly in [0, 1).
uniformDouble01M :: RandomGen g => g -> (Double, g)

-- | Generate a Double uniformly in (0, 1).
uniformDoublePositive01M :: RandomGen g => g -> (Double, g)

-- | Generate a Float uniformly in [0, 1).
uniformFloat01M :: RandomGen g => g -> (Float, g)

-- | Generate a Float uniformly in (0, 1).
uniformFloatPositive01M :: RandomGen g => g -> (Float, g)

-- | Generate uniformly distributed bytes.
uniformByteStringM :: RandomGen g => Int -> g -> (ByteString, g)

-- | Generate normally distributed value.
normalM :: RandomGen g => Double -> Double -> g -> (Double, g)

-- | Generate pair of independent normal values (Box-Muller).
normalPair :: RandomGen g => g -> ((Double, Double), g)

-- | Generate standard normal (mean=0, stddev=1).
standardNormal :: RandomGen g => g -> (Double, g)

-- | Generate exponentially distributed value.
exponentialM :: RandomGen g => Double -> g -> (Double, g)

-- | Generate gamma distributed value.
gammaM :: RandomGen g => Double -> Double -> g -> (Double, g)

-- | Generate beta distributed value.
betaM :: RandomGen g => Double -> Double -> g -> (Double, g)

-- | Generate Poisson distributed value.
poissonM :: RandomGen g => Double -> g -> (Int, g)

-- | Generate Bernoulli distributed value.
bernoulliM :: RandomGen g => Double -> g -> (Bool, g)

-- | Sample from categorical distribution.
categoricalM :: RandomGen g => [Double] -> g -> (Int, g)

-- | Shuffle a list randomly.
shuffle :: RandomGen g => [a] -> g -> ([a], g)

-- | Shuffle in IO.
shuffleM :: [a] -> IO [a]

-- | Sample n elements without replacement.
sample :: RandomGen g => Int -> [a] -> g -> ([a], g)

-- | Sample in IO.
sampleM :: Int -> [a] -> IO [a]

-- | Split generator into n independent generators.
splitN :: RandomGen g => Int -> g -> [g]

-- | Generate random tensor with values in [0, 1).
randomTensor :: Shape -> StdGen -> (Tensor F64, StdGen)

-- | Generate random tensor with values in range.
randomTensorR :: (Double, Double) -> Shape -> StdGen -> (Tensor F64, StdGen)

-- | Generate normally distributed tensor.
normalTensor :: Double -> Double -> Shape -> StdGen -> (Tensor F64, StdGen)

-- | Generate uniformly distributed tensor.
uniformTensor :: Shape -> StdGen -> (Tensor F64, StdGen)

-- | Seed for reproducible random sequences.
data Seed

-- | Create seed from integer.
seedFromInteger :: Integer -> Seed

-- | Convert seed to integer.
seedToInteger :: Seed -> Integer

-- | Create seed from bytes.
seedFromByteString :: ByteString -> Seed

-- | Convert seed to bytes.
seedToByteString :: Seed -> ByteString

-- | Random monad (state monad with generator).
type Rand g = RandT g Identity

-- | Random monad transformer.
data RandT g m a

-- | Run Rand computation.
runRand :: Rand g a -> g -> (a, g)

-- | Run RandT computation.
runRandT :: RandT g m a -> g -> m (a, g)

-- | Evaluate Rand, discarding final generator.
evalRand :: Rand g a -> g -> a

-- | Evaluate RandT, discarding final generator.
evalRandT :: Monad m => RandT g m a -> g -> m a

-- | Execute Rand, returning final generator.
execRand :: Rand g a -> g -> g

-- | Execute RandT, returning final generator.
execRandT :: Monad m => RandT g m a -> g -> m g

-- | Lift pure random function into Rand.
liftRand :: (g -> (a, g)) -> Rand g a

-- | Lift random function into RandT.
liftRandT :: (g -> m (a, g)) -> RandT g m a

-- | Class for monads with random generation capability.
class Monad m => MonadRandom m where
  getRandomM :: Random a => m a
  getRandomRM :: Random a => (a, a) -> m a

-- Phantom types for shape
type Shape = [Int]

-- This is a specification file.
-- Actual implementation provided by the compiler.
