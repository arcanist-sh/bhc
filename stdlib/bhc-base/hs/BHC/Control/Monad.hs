-- |
-- Module      : BHC.Control.Monad
-- Description : Monad class and operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Control.Monad (
    -- * Functor and Monad classes
    Functor(..),
    Applicative(..),
    Monad(..),
    MonadFail(..),
    
    -- * Functions
    -- ** Naming conventions
    -- | Functions with @M@ suffix work in a monad.
    -- Functions with @_@ suffix discard the result.
    
    -- ** Basic functions
    (=<<),
    (>=>), (<=<),
    forever,
    void,
    
    -- ** Generalisations of list functions
    join,
    msum, mfilter,
    filterM, mapAndUnzipM,
    zipWithM, zipWithM_,
    foldM, foldM_,
    replicateM, replicateM_,
    
    -- ** Conditional execution
    guard,
    when, unless,
    
    -- ** Monadic lifting
    liftM, liftM2, liftM3, liftM4, liftM5,
    ap,
    (<$!>),
    
    -- * MonadPlus
    MonadPlus(..),
) where

import BHC.Prelude

-- | Strict version of '<$>'.
(<$!>) :: Monad m => (a -> b) -> m a -> m b
f <$!> m = m >>= \x -> let z = f x in z `seq` return z
infixl 4 <$!>

-- | Lift a function to a monad.
liftM :: Monad m => (a -> b) -> m a -> m b
liftM f m = m >>= return . f

-- | Lift a binary function.
liftM2 :: Monad m => (a -> b -> c) -> m a -> m b -> m c
liftM2 f m1 m2 = do
    x1 <- m1
    x2 <- m2
    return (f x1 x2)

-- | Lift a ternary function.
liftM3 :: Monad m => (a -> b -> c -> d) -> m a -> m b -> m c -> m d
liftM3 f m1 m2 m3 = do
    x1 <- m1
    x2 <- m2
    x3 <- m3
    return (f x1 x2 x3)

-- | Lift a quaternary function.
liftM4 :: Monad m => (a -> b -> c -> d -> e) -> m a -> m b -> m c -> m d -> m e
liftM4 f m1 m2 m3 m4 = do
    x1 <- m1
    x2 <- m2
    x3 <- m3
    x4 <- m4
    return (f x1 x2 x3 x4)

-- | Lift a quinary function.
liftM5 :: Monad m => (a -> b -> c -> d -> e -> f) -> m a -> m b -> m c -> m d -> m e -> m f
liftM5 f m1 m2 m3 m4 m5 = do
    x1 <- m1
    x2 <- m2
    x3 <- m3
    x4 <- m4
    x5 <- m5
    return (f x1 x2 x3 x4 x5)

-- | Combine MonadPlus values.
msum :: (Foldable t, MonadPlus m) => t (m a) -> m a
msum = foldr mplus mzero

-- | Filter with a monadic predicate.
mfilter :: MonadPlus m => (a -> Bool) -> m a -> m a
mfilter p ma = do
    a <- ma
    if p a then return a else mzero
