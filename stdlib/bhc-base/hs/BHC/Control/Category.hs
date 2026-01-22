-- |
-- Module      : BHC.Control.Category
-- Description : Categories
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

{-# LANGUAGE PolyKinds #-}

module BHC.Control.Category (
    Category(..),
    (<<<), (>>>),
) where

import BHC.Prelude (id, (.))

-- | A category.
class Category cat where
    -- | Identity morphism
    id :: cat a a
    -- | Morphism composition
    (.) :: cat b c -> cat a b -> cat a c

instance Category (->) where
    id = BHC.Prelude.id
    (.) = (BHC.Prelude..)

-- | Right-to-left composition
(<<<) :: Category cat => cat b c -> cat a b -> cat a c
(<<<) = (.)
infixr 1 <<<

-- | Left-to-right composition
(>>>) :: Category cat => cat a b -> cat b c -> cat a c
f >>> g = g . f
infixr 1 >>>
