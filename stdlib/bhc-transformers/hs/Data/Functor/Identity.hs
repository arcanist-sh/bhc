-- |
-- Module      : Data.Functor.Identity
-- Description : The identity functor and monad
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005): ordinary Haskell,
-- compiled like any user module.
module Data.Functor.Identity
  ( Identity (..)
  ) where

-- | The trivial monad: a computation that just yields its value.
newtype Identity a = Identity { runIdentity :: a }

instance Functor Identity where
  fmap f (Identity a) = Identity (f a)

instance Applicative Identity where
  pure = Identity
  Identity f <*> Identity a = Identity (f a)

instance Monad Identity where
  Identity a >>= k = k a
