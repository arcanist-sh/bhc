-- |
-- Module      : Data.Semigroup
-- Description : Semigroup newtype wrappers
-- License     : BSD-3-Clause
-- Stability   : experimental
--
-- 'Min', 'Max' and the plain 'First'/'Last' semigroups, plus the wrappers
-- shared with "Data.Monoid".
--
-- Bundled with the compiler and compiled with the program (#52).
module Data.Semigroup
  ( Semigroup(..), Monoid(..)
  , Sum(..), Product(..), All(..), Any(..)
  , Min(..), Max(..), First(..), Last(..)
  ) where

import Data.Semigroup.Internal (Sum(..), Product(..), All(..), Any(..))

-- | The smaller of two values.
newtype Min a = Min { getMin :: a } deriving (Eq, Ord, Show)

instance Ord a => Semigroup (Min a) where
  Min a <> Min b = Min (min a b)

-- | The larger of two values.
newtype Max a = Max { getMax :: a } deriving (Eq, Ord, Show)

instance Ord a => Semigroup (Max a) where
  Max a <> Max b = Max (max a b)

-- | The leftmost value.
newtype First a = First { getFirst :: a } deriving (Eq, Ord, Show)

instance Semigroup (First a) where
  a <> _ = a

-- | The rightmost value.
newtype Last a = Last { getLast :: a } deriving (Eq, Ord, Show)

instance Semigroup (Last a) where
  _ <> b = b
