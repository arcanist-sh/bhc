-- |
-- Module      : BHC.Control.Applicative
-- Description : Applicative functors
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Control.Applicative (
    -- * Applicative functors
    Applicative(..),
    
    -- * Alternatives
    Alternative(..),
    
    -- * Utility functions
    (<$>), (<$),
    (<**>),
    liftA, liftA2, liftA3,
    optional,
    asum,
    
    -- * Const functor
    Const(..),
    getConst,
    
    -- * ZipList
    ZipList(..),
) where

import BHC.Prelude

-- | Flip of '<*>'.
(<**>) :: Applicative f => f a -> f (a -> b) -> f b
(<**>) = liftA2 (\a f -> f a)
infixl 4 <**>

-- | Zero or one.
optional :: Alternative f => f a -> f (Maybe a)
optional v = Just <$> v <|> pure Nothing

-- | Combine alternatives.
asum :: (Foldable t, Alternative f) => t (f a) -> f a
asum = foldr (<|>) empty

-- | Constant functor.
newtype Const a b = Const { getConst :: a }
    deriving (Eq, Ord, Show, Read, Bounded)

instance Functor (Const a) where
    fmap _ (Const x) = Const x

instance Monoid a => Applicative (Const a) where
    pure _ = Const mempty
    Const f <*> Const x = Const (f <> x)

-- | Lists with zippy Applicative.
newtype ZipList a = ZipList { getZipList :: [a] }
    deriving (Eq, Ord, Show, Read)

instance Functor ZipList where
    fmap f (ZipList xs) = ZipList (map f xs)

instance Applicative ZipList where
    pure x = ZipList (repeat x)
    ZipList fs <*> ZipList xs = ZipList (zipWith ($) fs xs)

instance Alternative ZipList where
    empty = ZipList []
    ZipList xs <|> ZipList ys = ZipList (xs ++ ys)
