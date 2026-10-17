{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
-- |
-- Module      : Control.Monad.State.Class
-- Description : The MonadState class
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.State.Class
  ( MonadState (..)
  , modify
  , modify'
  , gets
  ) where

import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Except (ExceptT)
import Control.Monad.Trans.Maybe (MaybeT)
import Control.Monad.Trans.Reader (ReaderT)
import qualified Control.Monad.Trans.State.Strict as S
import Control.Monad.Trans.Writer.Strict (WriterT)

-- | Monads with a state of type @s@.
class Monad m => MonadState s m | m -> s where
  -- | The current state.
  get :: m s
  -- | Replace the state.
  put :: s -> m ()
  -- | Embed a state transition.
  state :: (s -> (a, s)) -> m a
  state f = get >>= \s -> let (a, s') = f s in put s' >> return a

-- | Apply a function to the state.
modify :: MonadState s m => (s -> s) -> m ()
modify f = state (\s -> ((), f s))

-- | Apply a function to the state, forcing the new state.
modify' :: MonadState s m => (s -> s) -> m ()
modify' f = get >>= \s -> let s' = f s in s' `seq` put s'

-- | A projection of the state.
gets :: MonadState s m => (s -> a) -> m a
gets f = get >>= \s -> return (f s)

instance Monad m => MonadState s (S.StateT s m) where
  get = S.get
  put = S.put
  state = S.state

instance MonadState s m => MonadState s (ReaderT r m) where
  get = lift get
  put s = lift (put s)
  state f = lift (state f)

instance MonadState s m => MonadState s (ExceptT e m) where
  get = lift get
  put s = lift (put s)
  state f = lift (state f)

instance (Monoid w, MonadState s m) => MonadState s (WriterT w m) where
  get = lift get
  put s = lift (put s)
  state f = lift (state f)

instance MonadState s m => MonadState s (MaybeT m) where
  get = lift get
  put s = lift (put s)
  state f = lift (state f)
