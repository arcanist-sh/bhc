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

-- ------------------------------------------------------------
-- Basic combinators (re-exported from Prelude)
-- ------------------------------------------------------------

-- id :: a -> a
-- const :: a -> b -> a
-- (.) :: (b -> c) -> (a -> b) -> a -> c
-- flip :: (a -> b -> c) -> b -> a -> c
-- ($) :: (a -> b) -> a -> b
-- (&) :: a -> (a -> b) -> b
-- on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
-- fix :: (a -> a) -> a

-- ------------------------------------------------------------
-- Apply helpers
-- ------------------------------------------------------------

-- | /O(1)/. Conditionally apply a function.
-- @applyWhen True f x = f x@, @applyWhen False f x = x@.
--
-- >>> applyWhen True (+1) 5
-- 6
-- >>> applyWhen False (+1) 5
-- 5
--
-- Useful for optional transformations:
--
-- >>> let addPrefix flag = applyWhen flag ("prefix_" ++)
-- >>> addPrefix True "name"
-- "prefix_name"
applyWhen :: Bool -> (a -> a) -> a -> a
applyWhen True  f x = f x
applyWhen False _ x = x
