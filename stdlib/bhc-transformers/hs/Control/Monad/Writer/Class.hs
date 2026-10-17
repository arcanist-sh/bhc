{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
-- |
-- Module      : Control.Monad.Writer.Class
-- Description : The MonadWriter class
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.Writer.Class
  ( MonadWriter (..)
  , listens
  , censor
  ) where

import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Except (ExceptT (..))
import Control.Monad.Trans.Maybe (MaybeT (..))
import Control.Monad.Trans.Reader (ReaderT (..), mapReaderT)
import Control.Monad.Trans.State.Strict (StateT (..))
import qualified Control.Monad.Trans.Writer.Strict as W

-- | Monads that accumulate output of type @w@.
class (Monoid w, Monad m) => MonadWriter w m | m -> w where
  -- | Embed a result and its output.
  writer :: (a, w) -> m a
  writer (a, w) = tell w >> return a
  -- | Produce output.
  tell :: w -> m ()
  -- | Also return the output a computation produced.
  listen :: m a -> m (a, w)
  -- | Apply the function a computation returns to its output.
  pass :: m (a, w -> w) -> m a

-- | Also return a projection of the output.
listens :: MonadWriter w m => (w -> b) -> m a -> m (a, b)
listens f m = listen m >>= \(a, w) -> return (a, f w)

-- | Apply a function to a computation's output.
censor :: MonadWriter w m => (w -> w) -> m a -> m a
censor f m = pass (m >>= \a -> return (a, f))

instance (Monoid w, Monad m) => MonadWriter w (W.WriterT w m) where
  writer = W.writer
  tell = W.tell
  listen = W.listen
  pass = W.pass

instance MonadWriter w m => MonadWriter w (StateT s m) where
  tell w = lift (tell w)
  listen m = StateT (\s -> listen (runStateT m s) >>= \((a, s'), w) -> return ((a, w), s'))
  pass m = StateT (\s -> pass (runStateT m s >>= \((a, f), s') -> return ((a, s'), f)))

instance MonadWriter w m => MonadWriter w (ReaderT r m) where
  tell w = lift (tell w)
  listen = mapReaderT listen
  pass = mapReaderT pass

instance MonadWriter w m => MonadWriter w (ExceptT e m) where
  tell w = lift (tell w)
  listen (ExceptT m) = ExceptT (listen m >>= \(ea, w) -> return (fmap (\a -> (a, w)) ea))
  pass (ExceptT m) = ExceptT (pass (m >>= \ea -> return (case ea of
    Left e -> (Left e, id)
    Right (a, f) -> (Right a, f))))

instance MonadWriter w m => MonadWriter w (MaybeT m) where
  tell w = lift (tell w)
  listen (MaybeT m) = MaybeT (listen m >>= \(ma, w) -> return (fmap (\a -> (a, w)) ma))
  pass (MaybeT m) = MaybeT (pass (m >>= \ma -> return (case ma of
    Nothing -> (Nothing, id)
    Just (a, f) -> (Just a, f))))
