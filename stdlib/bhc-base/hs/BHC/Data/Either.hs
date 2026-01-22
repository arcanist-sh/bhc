-- |
-- Module      : BHC.Data.Either
-- Description : The Either type for error handling
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The 'Either' type represents values with two possibilities.

module BHC.Data.Either (
    Either(..),
    
    -- * Case analysis
    either,
    
    -- * Querying
    isLeft,
    isRight,
    
    -- * Extraction
    fromLeft,
    fromRight,
    
    -- * Partitioning
    lefts,
    rights,
    partitionEithers,
) where

import BHC.Prelude (Either(..), Bool(..), either)

-- | Return 'True' if 'Left'.
isLeft :: Either a b -> Bool
isLeft (Left _) = True
isLeft _        = False

-- | Return 'True' if 'Right'.
isRight :: Either a b -> Bool
isRight (Right _) = True
isRight _         = False

-- | Extract from 'Left' with default.
fromLeft :: a -> Either a b -> a
fromLeft _ (Left x) = x
fromLeft d _        = d

-- | Extract from 'Right' with default.
fromRight :: b -> Either a b -> b
fromRight _ (Right x) = x
fromRight d _         = d

-- | Extract all 'Left' values.
lefts :: [Either a b] -> [a]
lefts = foldr go []
  where go (Left x)  acc = x : acc
        go (Right _) acc = acc

-- | Extract all 'Right' values.
rights :: [Either a b] -> [b]
rights = foldr go []
  where go (Left _)  acc = acc
        go (Right x) acc = x : acc

-- | Partition into 'Left' and 'Right'.
partitionEithers :: [Either a b] -> ([a], [b])
partitionEithers = foldr go ([], [])
  where go (Left x)  (ls, rs) = (x:ls, rs)
        go (Right x) (ls, rs) = (ls, x:rs)
