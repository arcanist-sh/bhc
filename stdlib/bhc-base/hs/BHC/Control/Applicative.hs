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

-- ------------------------------------------------------------
-- Utility functions
-- ------------------------------------------------------------

-- | /O(1)/. Flip of '<*>'. Apply a wrapped function to a wrapped value
-- where the arguments are reversed.
--
-- >>> Just 5 <**> Just (+1)
-- Just 6
(<**>) :: Applicative f => f a -> f (a -> b) -> f b
(<**>) = liftA2 (\a f -> f a)
infixl 4 <**>

-- | /O(1)/. Zero or one. Turns an 'Alternative' into an optional value.
--
-- >>> optional (Just 5)
-- Just (Just 5)
-- >>> optional Nothing :: Maybe (Maybe Int)
-- Just Nothing
--
-- Commonly used with parser combinators.
optional :: Alternative f => f a -> f (Maybe a)
optional v = Just <$> v <|> pure Nothing

-- | /O(n)/. Combine alternatives using '<|>'.
--
-- >>> asum [Nothing, Just 1, Just 2]
-- Just 1
-- >>> asum [[], [1, 2], [3]]
-- [1,2,3]
asum :: (Foldable t, Alternative f) => t (f a) -> f a
asum = foldr (<|>) empty

-- ------------------------------------------------------------
-- Const functor
-- ------------------------------------------------------------

-- | The constant functor. Ignores the second type parameter.
-- Useful for implementing 'Traversable' instances.
--
-- >>> fmap (+1) (Const "hello")
-- Const "hello"
-- >>> getConst (Const "hello" <*> Const " world")
-- "hello world"
newtype Const a b = Const { getConst :: a }
    deriving (Eq, Ord, Show, Read, Bounded)

instance Functor (Const a) where
    fmap _ (Const x) = Const x

instance Monoid a => Applicative (Const a) where
    pure _ = Const mempty
    Const f <*> Const x = Const (f <> x)

-- ------------------------------------------------------------
-- ZipList
-- ------------------------------------------------------------

-- | Lists with a zippy 'Applicative'. Unlike the default list instance,
-- @ZipList@ applies functions elementwise.
--
-- >>> ZipList [(+1), (*2)] <*> ZipList [1, 2, 3]
-- ZipList {getZipList = [2,4]}
--
-- Compare with regular lists:
--
-- >>> [(+1), (*2)] <*> [1, 2, 3]
-- [2,3,4,2,4,6]
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
