-- |
-- Module      : Data.Monoid
-- Description : Monoid newtype wrappers
-- License     : BSD-3-Clause
-- Stability   : experimental
--
-- 'Sum', 'Product', 'All', 'Any' and the 'Maybe'-wrapping 'First'/'Last',
-- with their 'Semigroup' and 'Monoid' instances. ('Endo' is still a builtin.)
--
-- Bundled with the compiler and compiled with the program (#52).
module Data.Monoid
  ( Semigroup(..), Monoid(..)
  , Sum(..), Product(..), All(..), Any(..)
  , First(..), Last(..)
  ) where

import Data.Semigroup.Internal (Sum(..), Product(..), All(..), Any(..))

-- | The leftmost 'Just'.
newtype First a = First { getFirst :: Maybe a } deriving (Eq, Ord, Show)

instance Semigroup (First a) where
  First Nothing <> b = b
  a <> _ = a

instance Monoid (First a) where
  mempty = First Nothing

-- | The rightmost 'Just'.
newtype Last a = Last { getLast :: Maybe a } deriving (Eq, Ord, Show)

instance Semigroup (Last a) where
  a <> Last Nothing = a
  _ <> b = b

instance Monoid (Last a) where
  mempty = Last Nothing
