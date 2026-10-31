-- |
-- Module      : Data.Semigroup.Internal
-- Description : Newtype wrappers shared by Data.Monoid and Data.Semigroup
-- License     : BSD-3-Clause
-- Stability   : experimental
--
-- The 'Semigroup'/'Monoid' newtypes that "Data.Monoid" and "Data.Semigroup"
-- both export. They live here so that neither public module has to import
-- the other: each defines its own 'First'/'Last' (a 'Maybe' wrapper in
-- "Data.Monoid", a plain one in "Data.Semigroup"), and a program importing
-- one of them never compiles the other's.
--
-- Bundled with the compiler and compiled with the program (#52).
module Data.Semigroup.Internal
  ( Sum(..), Product(..), All(..), Any(..), Dual(..)
  ) where

-- | Monoid under addition.
newtype Sum a = Sum { getSum :: a } deriving (Eq, Ord, Show)

instance Num a => Semigroup (Sum a) where
  Sum a <> Sum b = Sum (a + b)

instance Num a => Monoid (Sum a) where
  mempty = Sum 0

-- | Monoid under multiplication.
newtype Product a = Product { getProduct :: a } deriving (Eq, Ord, Show)

instance Num a => Semigroup (Product a) where
  Product a <> Product b = Product (a * b)

instance Num a => Monoid (Product a) where
  mempty = Product 1

-- | Boolean monoid under conjunction ('&&').
newtype All = All { getAll :: Bool } deriving (Eq, Ord, Show)

instance Semigroup All where
  All a <> All b = All (a && b)

instance Monoid All where
  mempty = All True

-- | Boolean monoid under disjunction ('||').
newtype Any = Any { getAny :: Bool } deriving (Eq, Ord, Show)

instance Semigroup Any where
  Any a <> Any b = Any (a || b)

instance Monoid Any where
  mempty = Any False

-- | The dual of a monoid: '<>' with its arguments flipped.
newtype Dual a = Dual { getDual :: a } deriving (Eq, Ord, Show)

instance Semigroup a => Semigroup (Dual a) where
  Dual a <> Dual b = Dual (b <> a)

instance Monoid a => Monoid (Dual a) where
  mempty = Dual mempty
