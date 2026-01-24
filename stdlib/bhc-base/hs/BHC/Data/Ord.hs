-- |
-- Module      : BHC.Data.Ord
-- Description : Ordering and comparison
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Data.Ord (
    Ord(..),
    Ordering(..),
    Down(..),
    comparing,
    clamp,
) where

import BHC.Prelude (Ord(..), Ordering(..), on)

-- ------------------------------------------------------------
-- Ordering helpers
-- ------------------------------------------------------------

-- | Wrapper for reverse ordering. Useful for sorting in descending order.
--
-- >>> import Data.List (sortOn)
-- >>> sortOn Down [1, 3, 2]
-- [3,2,1]
newtype Down a = Down { getDown :: a }
    deriving (Eq, Show, Read)

instance Ord a => Ord (Down a) where
    compare (Down x) (Down y) = compare y x

-- | /O(1)/. Compare two values by first applying a key function.
-- Useful for sorting by a derived value.
--
-- >>> comparing length "hi" "hello"
-- LT
-- >>> comparing fst (1, "a") (2, "b")
-- LT
comparing :: Ord b => (a -> b) -> a -> a -> Ordering
comparing f = compare `on` f

-- | /O(1)/. Clamp a value to a range @(lo, hi)@.
-- Returns @lo@ if @x < lo@, @hi@ if @x > hi@, otherwise @x@.
--
-- >>> clamp (0, 10) 5
-- 5
-- >>> clamp (0, 10) (-5)
-- 0
-- >>> clamp (0, 10) 15
-- 10
clamp :: Ord a => (a, a) -> a -> a
clamp (lo, hi) x
    | x < lo    = lo
    | x > hi    = hi
    | otherwise = x
