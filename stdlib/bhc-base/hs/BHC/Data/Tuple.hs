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

-- | Extract first element of 3-tuple.
fst3 :: (a, b, c) -> a
fst3 (x, _, _) = x

-- | Extract second element of 3-tuple.
snd3 :: (a, b, c) -> b
snd3 (_, y, _) = y

-- | Extract third element of 3-tuple.
thd3 :: (a, b, c) -> c
thd3 (_, _, z) = z
