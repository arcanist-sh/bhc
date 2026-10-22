-- |
-- Module      : Control.Monad.Trans.Maybe
-- Description : The Maybe monad transformer
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005).
module Control.Monad.Trans.Maybe
  ( MaybeT (..)
  , mapMaybeT
  , hoistMaybe
  ) where

import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (MonadTrans (..))

-- | A computation that may produce no result.
newtype MaybeT m a = MaybeT { runMaybeT :: m (Maybe a) }

instance Functor m => Functor (MaybeT m) where
  fmap f (MaybeT m) = MaybeT (fmap (fmap f) m)

instance Monad m => Applicative (MaybeT m) where
  pure a = MaybeT (return (Just a))
  MaybeT mf <*> MaybeT mx = MaybeT (mf >>= \mf' -> case mf' of
    Nothing -> return Nothing
    Just f -> mx >>= \mx' -> case mx' of
      Nothing -> return Nothing
      Just x -> return (Just (f x)))

instance Monad m => Monad (MaybeT m) where
  MaybeT m >>= k = MaybeT (m >>= \ma -> case ma of
    Nothing -> return Nothing
    Just a -> runMaybeT (k a))

instance MonadTrans MaybeT where
  lift m = MaybeT (m >>= \a -> return (Just a))

instance MonadIO m => MonadIO (MaybeT m) where
  liftIO io = lift (liftIO io)

-- | Transform the inner action.
mapMaybeT :: (m (Maybe a) -> n (Maybe b)) -> MaybeT m a -> MaybeT n b
mapMaybeT f (MaybeT m) = MaybeT (f m)

-- | Lift a 'Maybe' into the transformer.
hoistMaybe :: Monad m => Maybe a -> MaybeT m a
hoistMaybe ma = MaybeT (return ma)
