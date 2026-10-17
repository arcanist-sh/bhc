-- |
-- Module      : Control.Monad.Trans.Writer.Strict
-- Description : The writer monad transformer
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005). The overloaded
-- operations live in "Control.Monad.Writer.Class".
module Control.Monad.Trans.Writer.Strict
  ( -- * The Writer monad
    Writer
  , runWriter
  , execWriter
    -- * The WriterT transformer
  , WriterT (..)
  , execWriterT
  , mapWriterT
    -- * Writer operations
  , writer
  , tell
  , listen
  , listens
  , pass
  , censor
  ) where

import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (MonadTrans (..))
import Data.Functor.Identity (Identity (..))

-- | A computation that accumulates output of type @w@.
newtype WriterT w m a = WriterT { runWriterT :: m (a, w) }

-- | A writer monad over 'Identity'.
type Writer w = WriterT w Identity

instance Functor m => Functor (WriterT w m) where
  fmap f (WriterT m) = WriterT (fmap (\(a, w) -> (f a, w)) m)

instance (Monoid w, Monad m) => Applicative (WriterT w m) where
  pure a = WriterT (return (a, mempty))
  WriterT mf <*> WriterT mx = WriterT (mf >>= \(f, w1) -> mx >>= \(x, w2) -> return (f x, w1 <> w2))

instance (Monoid w, Monad m) => Monad (WriterT w m) where
  WriterT m >>= k = WriterT (m >>= \(a, w1) -> runWriterT (k a) >>= \(b, w2) -> return (b, w1 <> w2))

instance Monoid w => MonadTrans (WriterT w) where
  lift m = WriterT (m >>= \a -> return (a, mempty))

instance (Monoid w, MonadIO m) => MonadIO (WriterT w m) where
  liftIO io = lift (liftIO io)

-- | Run a 'Writer' computation.
runWriter :: Writer w a -> (a, w)
runWriter m = runIdentity (runWriterT m)

-- | The output of a 'Writer' computation.
execWriter :: Writer w a -> w
execWriter m = snd (runWriter m)

-- | The output of a 'WriterT' computation.
execWriterT :: Monad m => WriterT w m a -> m w
execWriterT m = runWriterT m >>= \(_, w) -> return w

-- | Transform the inner action.
mapWriterT :: (m (a, w) -> n (b, w')) -> WriterT w m a -> WriterT w' n b
mapWriterT f (WriterT m) = WriterT (f m)

-- | Embed a result and its output.
writer :: Monad m => (a, w) -> WriterT w m a
writer aw = WriterT (return aw)

-- | Produce output.
tell :: Monad m => w -> WriterT w m ()
tell w = WriterT (return ((), w))

-- | Also return the output a computation produced.
listen :: Monad m => WriterT w m a -> WriterT w m (a, w)
listen (WriterT m) = WriterT (m >>= \(a, w) -> return ((a, w), w))

-- | Also return a projection of the output.
listens :: Monad m => (w -> b) -> WriterT w m a -> WriterT w m (a, b)
listens f (WriterT m) = WriterT (m >>= \(a, w) -> return ((a, f w), w))

-- | Apply the function a computation returns to its output.
pass :: Monad m => WriterT w m (a, w -> w) -> WriterT w m a
pass (WriterT m) = WriterT (m >>= \((a, f), w) -> return (a, f w))

-- | Apply a function to a computation's output.
censor :: Monad m => (w -> w) -> WriterT w m a -> WriterT w m a
censor f (WriterT m) = WriterT (m >>= \(a, w) -> return (a, f w))
