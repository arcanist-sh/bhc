-- |
-- Module      : BHC.Data.Function
-- Description : Function combinators
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Data.Function (
    -- * Basic
    id, const, (.), flip, ($), (&),
    
    -- * Composition
    on,
    
    -- * Fixed point
    fix,
    
    -- * Apply helpers
    applyWhen,
) where

import BHC.Prelude (id, const, (.), flip, ($), (&), on, fix, Bool(..))

-- | Conditionally apply a function.
applyWhen :: Bool -> (a -> a) -> a -> a
applyWhen True  f x = f x
applyWhen False _ x = x
