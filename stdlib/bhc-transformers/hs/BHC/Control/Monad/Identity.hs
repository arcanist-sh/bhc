-- |
-- Module      : BHC.Control.Monad.Identity
-- Description : Identity monad and transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The identity monad and identity monad transformer.
--
-- The identity monad is a monad that does not embody any computational
-- strategy. It simply applies the bound function to its input without
-- any additional modification.
--
-- = Usage
--
-- @
-- import BHC.Control.Monad.Identity
--
-- -- Identity is the base case for transformer stacks
-- type MyMonad = StateT Int (ReaderT Config Identity)
--
-- runMyMonad :: MyMonad a -> Config -> Int -> (a, Int)
-- runMyMonad m config s = runIdentity (runReaderT (runStateT m s) config)
-- @

module BHC.Control.Monad.Identity
    ( -- * The Identity monad
      Identity(..)
    , runIdentity

      -- * The IdentityT monad transformer
    , IdentityT(..)
    , runIdentityT
    , mapIdentityT
    ) where

import BHC.Prelude
import BHC.Control.Monad.Trans

-- ============================================================
-- Identity monad
-- ============================================================

-- | The identity monad wraps a value without adding any effect.
--
-- Computation type: Simple function application.
-- Binding strategy: Applies the function directly.
-- Useful for: Unified treatment of parameterized and unparameterized code.
newtype Identity a = Identity { runIdentity :: a }
    deriving (Eq, Ord, Show)

instance Functor Identity where
    fmap f (Identity x) = Identity (f x)
    {-# INLINE fmap #-}

instance Applicative Identity where
    pure = Identity
    {-# INLINE pure #-}

    Identity f <*> Identity x = Identity (f x)
    {-# INLINE (<*>) #-}

    Identity _ *> Identity y = Identity y
    {-# INLINE (*>) #-}

    Identity x <* Identity _ = Identity x
    {-# INLINE (<*) #-}

instance Monad Identity where
    Identity x >>= f = f x
    {-# INLINE (>>=) #-}

instance Foldable Identity where
    foldMap f (Identity x) = f x
    {-# INLINE foldMap #-}

    foldr f z (Identity x) = f x z
    {-# INLINE foldr #-}

    foldl f z (Identity x) = f z x
    {-# INLINE foldl #-}

    length _ = 1
    {-# INLINE length #-}

    null _ = False
    {-# INLINE null #-}

instance Traversable Identity where
    traverse f (Identity x) = Identity <$> f x
    {-# INLINE traverse #-}

-- ============================================================
-- IdentityT monad transformer
-- ============================================================

-- | The identity monad transformer.
--
-- This transformer is equivalent to the underlying monad.
-- It is useful as a base case for transformer stacks or when
-- a transformer is required but no additional effect is needed.
newtype IdentityT m a = IdentityT { runIdentityT :: m a }
    deriving (Eq, Ord, Show)

instance Functor m => Functor (IdentityT m) where
    fmap f (IdentityT m) = IdentityT (fmap f m)
    {-# INLINE fmap #-}

instance Applicative m => Applicative (IdentityT m) where
    pure = IdentityT . pure
    {-# INLINE pure #-}

    IdentityT f <*> IdentityT x = IdentityT (f <*> x)
    {-# INLINE (<*>) #-}

    IdentityT m *> IdentityT n = IdentityT (m *> n)
    {-# INLINE (*>) #-}

    IdentityT m <* IdentityT n = IdentityT (m <* n)
    {-# INLINE (<*) #-}

instance Monad m => Monad (IdentityT m) where
    IdentityT m >>= k = IdentityT $ m >>= runIdentityT . k
    {-# INLINE (>>=) #-}

instance MonadTrans IdentityT where
    lift = IdentityT
    {-# INLINE lift #-}

instance MonadIO m => MonadIO (IdentityT m) where
    liftIO = IdentityT . liftIO
    {-# INLINE liftIO #-}

instance Foldable m => Foldable (IdentityT m) where
    foldMap f (IdentityT m) = foldMap f m
    {-# INLINE foldMap #-}

    foldr f z (IdentityT m) = foldr f z m
    {-# INLINE foldr #-}

    foldl f z (IdentityT m) = foldl f z m
    {-# INLINE foldl #-}

instance Traversable m => Traversable (IdentityT m) where
    traverse f (IdentityT m) = IdentityT <$> traverse f m
    {-# INLINE traverse #-}

-- | Transform the computation inside an 'IdentityT'.
--
-- ==== __Examples__
--
-- >>> mapIdentityT (fmap (+1)) (IdentityT [1,2,3])
-- IdentityT [2,3,4]
mapIdentityT :: (m a -> n b) -> IdentityT m a -> IdentityT n b
mapIdentityT f (IdentityT m) = IdentityT (f m)
{-# INLINE mapIdentityT #-}
