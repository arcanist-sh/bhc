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
--
-- Arrows generalize functions, allowing computations that have
-- additional structure (state, effects, etc.) while still supporting
-- composition.
--
-- ==== __Laws__
--
-- Arrows must satisfy the following laws:
--
-- @
-- arr id = id
-- arr (f . g) = arr f . arr g
-- first (arr f) = arr (first f)
-- first (f . g) = first f . first g
-- first f . arr (id *** g) = arr (id *** g) . first f
-- @
class Category a => Arrow a where
    -- | /O(1)/. Lift a pure function to an arrow.
    --
    -- >>> arr (+1) 5
    -- 6
    arr :: (b -> c) -> a b c

    -- | /O(1)/. Send the first component of the input through the arrow,
    -- and copy the rest unchanged to the output.
    --
    -- >>> first (+1) (5, "hello")
    -- (6,"hello")
    first :: a b c -> a (b, d) (c, d)

    -- | /O(1)/. Like 'first', but on the second component.
    --
    -- >>> second (+1) ("hello", 5)
    -- ("hello",6)
    second :: a b c -> a (d, b) (d, c)
    second f = arr swap >>> first f >>> arr swap
      where swap (x, y) = (y, x)

    -- | /O(1)/. Split the input between two arrows, combining their output.
    --
    -- >>> ((*2) *** (+10)) (3, 5)
    -- (6,15)
    (***) :: a b c -> a b' c' -> a (b, b') (c, c')
    f *** g = first f >>> second g

    -- | /O(1)/. Fanout: send input to both arrows, combining output.
    --
    -- >>> ((*2) &&& (+10)) 5
    -- (10,15)
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

-- | /O(1)/. The identity arrow. Equivalent to @arr id@.
--
-- >>> returnA 5
-- 5
returnA :: Arrow a => a b b
returnA = arr id

-- | /O(1)/. Precomposition with a pure function (left to right).
--
-- >>> ((*2) ^>> arr (+1)) 5
-- 11
(^>>) :: Arrow a => (b -> c) -> a c d -> a b d
f ^>> a = arr f >>> a
infixr 1 ^>>

-- | /O(1)/. Postcomposition with a pure function (left to right).
--
-- >>> (arr (*2) >>^ (+1)) 5
-- 11
(>>^) :: Arrow a => a b c -> (c -> d) -> a b d
a >>^ f = a >>> arr f
infixr 1 >>^

-- | /O(1)/. Precomposition with a pure function (right to left).
--
-- >>> (arr (+1) <<^ (*2)) 5
-- 11
(<<^) :: Arrow a => a c d -> (b -> c) -> a b d
a <<^ f = a <<< arr f
infixr 1 <<^

-- | /O(1)/. Postcomposition with a pure function (right to left).
--
-- >>> ((+1) ^<< arr (*2)) 5
-- 11
(^<<) :: Arrow a => (c -> d) -> a b c -> a b d
f ^<< a = arr f <<< a
infixr 1 ^<<

-- | Arrows with a zero (identity for '<+>').
--
-- @zeroArrow@ represents a computation that always fails.
class Arrow a => ArrowZero a where
    -- | The zero arrow. Always fails/produces no output.
    zeroArrow :: a b c

-- | Arrows with a plus operation (choice between arrows).
class Arrow a => ArrowPlus a where
    -- | /O(1)/. Choice: try the first arrow, then the second.
    --
    -- For Kleisli arrows over MonadPlus, this uses 'mplus'.
    (<+>) :: a b c -> a b c -> a b c

infixr 5 <+>

-- | Arrows that support choice (branching on 'Either').
--
-- ArrowChoice allows arrows to branch based on the structure
-- of their input, processing Left and Right values differently.
class Arrow a => ArrowChoice a where
    -- | /O(1)/. Feed marked inputs through the left arrow.
    -- Right values pass through unchanged.
    --
    -- >>> left (+1) (Left 5)
    -- Left 6
    -- >>> left (+1) (Right "hello")
    -- Right "hello"
    left :: a b c -> a (Either b d) (Either c d)

    -- | /O(1)/. Feed marked inputs through the right arrow.
    -- Left values pass through unchanged.
    --
    -- >>> right (+1) (Right 5)
    -- Right 6
    -- >>> right (+1) (Left "hello")
    -- Left "hello"
    right :: a b c -> a (Either d b) (Either d c)
    right f = arr mirror >>> left f >>> arr mirror
      where mirror (Left x) = Right x
            mirror (Right x) = Left x

    -- | /O(1)/. Split the input between two arrows based on 'Either'.
    --
    -- >>> ((*2) +++ (+10)) (Left 5)
    -- Left 10
    -- >>> ((*2) +++ (+10)) (Right 5)
    -- Right 15
    (+++) :: a b c -> a b' c' -> a (Either b b') (Either c c')
    f +++ g = left f >>> right g

    -- | /O(1)/. Fanin: merge two computations into one output type.
    --
    -- >>> ((*2) ||| (+10)) (Left 5)
    -- 10
    -- >>> ((*2) ||| (+10)) (Right 5)
    -- 15
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

-- | Arrows that support application (first-class arrows).
--
-- 'ArrowApply' is equivalent in power to 'Monad'. The 'app' combinator
-- allows an arrow to be applied to its argument within the arrow computation.
class Arrow a => ArrowApply a where
    -- | /O(1)/. Apply an arrow to its argument.
    --
    -- >>> app ((*2), 5)
    -- 10
    app :: a (a b c, b) c

instance ArrowApply (->) where
    app (f, x) = f x

-- | The monad induced by 'ArrowApply'.
--
-- Any 'ArrowApply' gives rise to a 'Monad' via this newtype wrapper.
newtype ArrowMonad a b = ArrowMonad (a () b)

instance Arrow a => Functor (ArrowMonad a) where
    fmap f (ArrowMonad m) = ArrowMonad (m >>> arr f)

instance Arrow a => Applicative (ArrowMonad a) where
    pure = ArrowMonad . arr . const
    ArrowMonad f <*> ArrowMonad x = ArrowMonad (f &&& x >>> arr (uncurry ($)))

instance ArrowApply a => Monad (ArrowMonad a) where
    ArrowMonad m >>= f = ArrowMonad $
        m >>> arr (\x -> let ArrowMonad h = f x in (h, ())) >>> app

-- | /O(1)/. Apply an arrow to the left of 'Either'.
--
-- This is useful when you have an 'ArrowApply' but need 'ArrowChoice'-like
-- behavior.
leftApp :: ArrowApply a => a b c -> a (Either b d) (Either c d)
leftApp f = arr ((\b -> (arr (\() -> b) >>> f >>> arr Left, ())) |||
                 (\d -> (arr (\() -> d) >>> arr Right, ()))) >>> app

-- | Arrows with a feedback loop (fixed-point combinator).
--
-- 'ArrowLoop' allows the output of an arrow to be fed back as input,
-- enabling recursive arrow computations.
--
-- ==== __Laws__
--
-- @
-- loop (arr f) = arr (\\ b -> fst (fix (\\ (c,d) -> f (b,d))))
-- loop (first f) = f
-- loop (f >>> arr (id *** k)) = loop (arr (id *** k) >>> f)
-- loop (loop f) = loop (arr unassoc >>> f >>> arr assoc)
-- @
class Arrow a => ArrowLoop a where
    -- | /O(1)/. Create a feedback loop.
    --
    -- The second component of the output is fed back as the second
    -- component of the input.
    loop :: a (b, d) (c, d) -> a b c

instance ArrowLoop (->) where
    loop f b = let (c, d) = f (b, d) in c

-- | Kleisli arrows of a monad.
--
-- 'Kleisli' wraps a monadic function @a -> m b@ as an arrow.
-- This allows monadic computations to be used with arrow combinators.
--
-- >>> runKleisli (Kleisli (\x -> Just (x + 1)) >>> Kleisli (\x -> Just (x * 2))) 5
-- Just 12
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
