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

-- | Reverse ordering wrapper.
newtype Down a = Down { getDown :: a }
    deriving (Eq, Show, Read)

instance Ord a => Ord (Down a) where
    compare (Down x) (Down y) = compare y x

-- | Compare by applying a function.
comparing :: Ord b => (a -> b) -> a -> a -> Ordering
comparing f = compare `on` f

-- | Clamp a value to a range.
clamp :: Ord a => (a, a) -> a -> a
clamp (lo, hi) x
    | x < lo    = lo
    | x > hi    = hi
    | otherwise = x
