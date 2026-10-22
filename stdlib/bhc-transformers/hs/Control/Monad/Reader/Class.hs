{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
-- |
-- Module      : Control.Monad.Reader.Class
-- Description : The MonadReader class
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.Reader.Class
  ( MonadReader (..)
  , asks
  ) where

import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Except (ExceptT (..), mapExceptT)
import Control.Monad.Trans.Maybe (MaybeT (..), mapMaybeT)
import qualified Control.Monad.Trans.Reader as R
import qualified Control.Monad.Trans.RWS.Strict as RWS
import Control.Monad.Trans.State.Strict (StateT (..), mapStateT)
import Control.Monad.Trans.Writer.Strict (WriterT (..), mapWriterT)

-- | Monads with an environment of type @r@.
class Monad m => MonadReader r m | m -> r where
  -- | The environment.
  ask :: m r
  -- | Run a computation in a modified environment.
  local :: (r -> r) -> m a -> m a
  -- | Embed a function of the environment.
  reader :: (r -> a) -> m a
  reader f = ask >>= \r -> return (f r)

-- | A projection of the environment.
asks :: MonadReader r m => (r -> a) -> m a
asks = reader

instance Monad m => MonadReader r (R.ReaderT r m) where
  ask = R.ask
  local = R.local
  reader = R.reader

instance MonadReader r m => MonadReader r (StateT s m) where
  ask = lift ask
  local f = mapStateT (local f)
  reader f = lift (reader f)

instance MonadReader r m => MonadReader r (ExceptT e m) where
  ask = lift ask
  local f = mapExceptT (local f)
  reader f = lift (reader f)

instance (Monoid w, MonadReader r m) => MonadReader r (WriterT w m) where
  ask = lift ask
  local f = mapWriterT (local f)
  reader f = lift (reader f)

instance MonadReader r m => MonadReader r (MaybeT m) where
  ask = lift ask
  local f = mapMaybeT (local f)
  reader f = lift (reader f)

instance (Monoid w, Monad m) => MonadReader r (RWS.RWST r w s m) where
  ask = RWS.ask
  local = RWS.local
  reader = RWS.reader
