-- |
-- Module      : BHC.Data.Tuple
-- Description : Tuple operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Data.Tuple (
    fst, snd,
    curry, uncurry,
    swap,
    -- * For 3-tuples
    fst3, snd3, thd3,
) where

import BHC.Prelude (fst, snd, curry, uncurry, swap)

-- ------------------------------------------------------------
-- 2-tuple operations (re-exported from Prelude)
-- ------------------------------------------------------------

-- fst :: (a, b) -> a
-- snd :: (a, b) -> b
-- curry :: ((a, b) -> c) -> a -> b -> c
-- uncurry :: (a -> b -> c) -> (a, b) -> c
-- swap :: (a, b) -> (b, a)

-- ------------------------------------------------------------
-- 3-tuple operations
-- ------------------------------------------------------------

-- | /O(1)/. Extract the first element of a 3-tuple.
--
-- >>> fst3 (1, 2, 3)
-- 1
fst3 :: (a, b, c) -> a
fst3 (x, _, _) = x

-- | /O(1)/. Extract the second element of a 3-tuple.
--
-- >>> snd3 (1, 2, 3)
-- 2
snd3 :: (a, b, c) -> b
snd3 (_, y, _) = y

-- | /O(1)/. Extract the third element of a 3-tuple.
--
-- >>> thd3 (1, 2, 3)
-- 3
thd3 :: (a, b, c) -> c
thd3 (_, _, z) = z
