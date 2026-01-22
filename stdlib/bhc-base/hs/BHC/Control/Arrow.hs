-- |
-- Module      : BHC.Control.Arrow
-- Description : Arrow abstraction
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Arrows are a generalization of monads providing
-- a notation for computations.

{-# LANGUAGE PolyKinds #-}

module BHC.Control.Arrow (
    -- * Arrow class
    Arrow(..),
    
    -- * Derived combinators
    returnA,
    (^>>), (>>^),
    (>>>), (<<<),
    (<<^), (^<<),
    
    -- * ArrowZero and ArrowPlus
    ArrowZero(..),
    ArrowPlus(..),
    
    -- * ArrowChoice
    ArrowChoice(..),
    
    -- * ArrowApply
    ArrowApply(..),
    ArrowMonad(..),
    leftApp,
    
    -- * ArrowLoop
    ArrowLoop(..),
    
    -- * Kleisli arrows
    Kleisli(..),
) where

import BHC.Prelude
import BHC.Control.Category

-- | The basic arrow class.
class Category a => Arrow a where
    -- | Lift a function to an arrow.
    arr :: (b -> c) -> a b c
    
    -- | Send the first component of the input through the arrow,
    -- and copy the rest unchanged to the output.
    first :: a b c -> a (b, d) (c, d)
    
    -- | Like 'first', but on the second component.
    second :: a b c -> a (d, b) (d, c)
    second f = arr swap >>> first f >>> arr swap
      where swap (x, y) = (y, x)
    
    -- | Split the input between two arrows, combining their output.
    (***) :: a b c -> a b' c' -> a (b, b') (c, c')
    f *** g = first f >>> second g
    
    -- | Fanout: send input to both arrows, combining output.
    (&&&) :: a b c -> a b c' -> a b (c, c')
    f &&& g = arr (\b -> (b, b)) >>> f *** g

infixr 3 ***
infixr 3 &&&

instance Arrow (->) where
    arr = id
    first f (x, y) = (f x, y)
    second f (x, y) = (x, f y)
    (f *** g) (x, y) = (f x, g y)
    (f &&& g) x = (f x, g x)

-- | The identity arrow.
returnA :: Arrow a => a b b
returnA = arr id

-- | Precomposition with a pure function.
(^>>) :: Arrow a => (b -> c) -> a c d -> a b d
f ^>> a = arr f >>> a
infixr 1 ^>>

-- | Postcomposition with a pure function.
(>>^) :: Arrow a => a b c -> (c -> d) -> a b d
a >>^ f = a >>> arr f
infixr 1 >>^

-- | Precomposition (right to left).
(<<^) :: Arrow a => a c d -> (b -> c) -> a b d
a <<^ f = a <<< arr f
infixr 1 <<^

-- | Postcomposition (right to left).
(^<<) :: Arrow a => (c -> d) -> a b c -> a b d
f ^<< a = arr f <<< a
infixr 1 ^<<

-- | Arrows with a zero.
class Arrow a => ArrowZero a where
    zeroArrow :: a b c

-- | Arrows with choice.
class Arrow a => ArrowPlus a where
    (<+>) :: a b c -> a b c -> a b c

infixr 5 <+>

-- | Arrows that support choice.
class Arrow a => ArrowChoice a where
    -- | Feed marked inputs through the left arrow.
    left :: a b c -> a (Either b d) (Either c d)
    
    -- | Feed marked inputs through the right arrow.
    right :: a b c -> a (Either d b) (Either d c)
    right f = arr mirror >>> left f >>> arr mirror
      where mirror (Left x) = Right x
            mirror (Right x) = Left x
    
    -- | Split the input between two arrows.
    (+++) :: a b c -> a b' c' -> a (Either b b') (Either c c')
    f +++ g = left f >>> right g
    
    -- | Fanin: merge two computations.
    (|||) :: a b d -> a c d -> a (Either b c) d
    f ||| g = f +++ g >>> arr untag
      where untag (Left x) = x
            untag (Right x) = x

infixr 2 +++
infixr 2 |||

instance ArrowChoice (->) where
    left f (Left x)  = Left (f x)
    left _ (Right y) = Right y
    right _ (Left x)  = Left x
    right f (Right y) = Right (f y)
    (f +++ g) (Left x)  = Left (f x)
    (f +++ g) (Right y) = Right (g y)
    (f ||| g) (Left x)  = f x
    (f ||| g) (Right y) = g y

-- | Arrows that support application.
class Arrow a => ArrowApply a where
    app :: a (a b c, b) c

instance ArrowApply (->) where
    app (f, x) = f x

-- | Monad induced by ArrowApply.
newtype ArrowMonad a b = ArrowMonad (a () b)

instance Arrow a => Functor (ArrowMonad a) where
    fmap f (ArrowMonad m) = ArrowMonad (m >>> arr f)

instance Arrow a => Applicative (ArrowMonad a) where
    pure = ArrowMonad . arr . const
    ArrowMonad f <*> ArrowMonad x = ArrowMonad (f &&& x >>> arr (uncurry ($)))

instance ArrowApply a => Monad (ArrowMonad a) where
    ArrowMonad m >>= f = ArrowMonad $
        m >>> arr (\x -> let ArrowMonad h = f x in (h, ())) >>> app

-- | Apply an arrow to the left of either.
leftApp :: ArrowApply a => a b c -> a (Either b d) (Either c d)
leftApp f = arr ((\b -> (arr (\() -> b) >>> f >>> arr Left, ())) |||
                 (\d -> (arr (\() -> d) >>> arr Right, ()))) >>> app

-- | Arrows with a loop.
class Arrow a => ArrowLoop a where
    loop :: a (b, d) (c, d) -> a b c

instance ArrowLoop (->) where
    loop f b = let (c, d) = f (b, d) in c

-- | Kleisli arrows of a monad.
newtype Kleisli m a b = Kleisli { runKleisli :: a -> m b }

instance Monad m => Category (Kleisli m) where
    id = Kleisli return
    Kleisli f . Kleisli g = Kleisli (\x -> g x >>= f)

instance Monad m => Arrow (Kleisli m) where
    arr f = Kleisli (return . f)
    first (Kleisli f) = Kleisli (\(x, y) -> f x >>= \x' -> return (x', y))
    second (Kleisli f) = Kleisli (\(x, y) -> f y >>= \y' -> return (x, y'))

instance Monad m => ArrowChoice (Kleisli m) where
    left (Kleisli f) = Kleisli $ \e -> case e of
        Left x  -> f x >>= return . Left
        Right y -> return (Right y)

instance Monad m => ArrowApply (Kleisli m) where
    app = Kleisli (\(Kleisli f, x) -> f x)

instance MonadPlus m => ArrowZero (Kleisli m) where
    zeroArrow = Kleisli (const mzero)

instance MonadPlus m => ArrowPlus (Kleisli m) where
    Kleisli f <+> Kleisli g = Kleisli (\x -> f x `mplus` g x)
