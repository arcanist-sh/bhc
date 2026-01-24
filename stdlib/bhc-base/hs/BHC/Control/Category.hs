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

-- ------------------------------------------------------------
-- Category class
-- ------------------------------------------------------------

-- | A category is a mathematical structure with identity and composition.
-- The function arrow @(->)@ is the canonical example.
--
-- Laws:
--
-- * @id . f = f@ (left identity)
-- * @f . id = f@ (right identity)
-- * @(f . g) . h = f . (g . h)@ (associativity)
class Category cat where
    -- | Identity morphism. For functions, this is @\x -> x@.
    id :: cat a a
    -- | Morphism composition. For functions, this is @(f . g) x = f (g x)@.
    (.) :: cat b c -> cat a b -> cat a c

instance Category (->) where
    id = BHC.Prelude.id
    (.) = (BHC.Prelude..)

-- ------------------------------------------------------------
-- Composition operators
-- ------------------------------------------------------------

-- | Right-to-left composition. Same as @(.)@.
--
-- >>> ((*2) <<< (+1)) 3
-- 8
(<<<) :: Category cat => cat b c -> cat a b -> cat a c
(<<<) = (.)
infixr 1 <<<

-- | Left-to-right composition. Flipped @(.)@.
-- Often more readable for pipelines.
--
-- >>> ((+1) >>> (*2)) 3
-- 8
-- >>> (show >>> length) 12345
-- 5
(>>>) :: Category cat => cat a b -> cat b c -> cat a c
f >>> g = g . f
infixr 1 >>>
