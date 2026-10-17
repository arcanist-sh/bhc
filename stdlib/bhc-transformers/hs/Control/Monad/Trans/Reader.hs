-- |
-- Module      : Control.Monad.Trans.Reader
-- Description : The reader monad transformer
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005). The overloaded
-- operations live in "Control.Monad.Reader.Class".
module Control.Monad.Trans.Reader
  ( -- * The Reader monad
    Reader
  , runReader
  , mapReader
  , withReader
    -- * The ReaderT transformer
  , ReaderT (..)
  , mapReaderT
  , withReaderT
    -- * Reader operations
  , ask
  , local
  , asks
  , reader
  ) where

import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (MonadTrans (..))
import Data.Functor.Identity (Identity (..))

-- | A computation that reads an environment of type @r@.
newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }

-- | A reader monad over 'Identity'.
type Reader r = ReaderT r Identity

instance Functor m => Functor (ReaderT r m) where
  fmap f (ReaderT g) = ReaderT (\r -> fmap f (g r))

instance Applicative m => Applicative (ReaderT r m) where
  pure a = ReaderT (\_ -> pure a)
  ReaderT mf <*> ReaderT mx = ReaderT (\r -> mf r <*> mx r)

instance Monad m => Monad (ReaderT r m) where
  ReaderT m >>= k = ReaderT (\r -> m r >>= \a -> runReaderT (k a) r)

instance MonadTrans (ReaderT r) where
  lift m = ReaderT (\_ -> m)

instance MonadIO m => MonadIO (ReaderT r m) where
  liftIO io = lift (liftIO io)

-- | Run a 'Reader' computation with an environment.
runReader :: Reader r a -> r -> a
runReader m r = runIdentity (runReaderT m r)

-- | Transform the result of a 'Reader' computation.
mapReader :: (a -> b) -> Reader r a -> Reader r b
mapReader f = mapReaderT (\m -> Identity (f (runIdentity m)))

-- | Transform the environment before running.
withReader :: (r' -> r) -> Reader r a -> Reader r' a
withReader = withReaderT

-- | Transform the inner action.
mapReaderT :: (m a -> n b) -> ReaderT r m a -> ReaderT r n b
mapReaderT f (ReaderT g) = ReaderT (\r -> f (g r))

-- | Transform the environment before running.
withReaderT :: (r' -> r) -> ReaderT r m a -> ReaderT r' m a
withReaderT f (ReaderT g) = ReaderT (\r -> g (f r))

-- | The environment.
ask :: Monad m => ReaderT r m r
ask = ReaderT return

-- | Run a computation in a modified environment.
local :: (r -> r) -> ReaderT r m a -> ReaderT r m a
local = withReaderT

-- | A projection of the environment.
asks :: Monad m => (r -> a) -> ReaderT r m a
asks f = ReaderT (\r -> return (f r))

-- | Embed a function of the environment.
reader :: Monad m => (r -> a) -> ReaderT r m a
reader = asks
