-- |
-- Module      : BHC.Data.Maybe
-- Description : The Maybe type and related operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The 'Maybe' type encapsulates an optional value.

module BHC.Data.Maybe (
    Maybe(..),
    
    -- * Querying
    maybe,
    isJust,
    isNothing,
    
    -- * Extraction
    fromMaybe,
    fromJust,
    
    -- * Conversion
    listToMaybe,
    maybeToList,
    catMaybes,
    mapMaybe,
) where

import BHC.Prelude (Maybe(..), Bool(..), maybe)

-- | Returns 'True' iff the argument is 'Just'.
isJust :: Maybe a -> Bool
isJust (Just _) = True
isJust Nothing  = False

-- | Returns 'True' iff the argument is 'Nothing'.
isNothing :: Maybe a -> Bool
isNothing Nothing = True
isNothing _       = False

-- | Extract the value with a default.
fromMaybe :: a -> Maybe a -> a
fromMaybe d Nothing  = d
fromMaybe _ (Just x) = x

-- | Extract the value. Throws error on 'Nothing'.
fromJust :: Maybe a -> a
fromJust (Just x) = x
fromJust Nothing  = error "Maybe.fromJust: Nothing"

-- | Convert a list to 'Maybe' using the first element.
listToMaybe :: [a] -> Maybe a
listToMaybe []    = Nothing
listToMaybe (x:_) = Just x

-- | Convert 'Maybe' to a list.
maybeToList :: Maybe a -> [a]
maybeToList Nothing  = []
maybeToList (Just x) = [x]

-- | Filter out 'Nothing' values.
catMaybes :: [Maybe a] -> [a]
catMaybes = mapMaybe id

-- | Map and filter in one pass.
mapMaybe :: (a -> Maybe b) -> [a] -> [b]
mapMaybe _ []     = []
mapMaybe f (x:xs) = case f x of
    Nothing -> mapMaybe f xs
    Just y  -> y : mapMaybe f xs
