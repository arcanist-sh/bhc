-- |
-- Module      : Control.Monad.Trans.State.Strict
-- Description : The state monad transformer
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005). The operations here
-- are specific to 'StateT'; the overloaded versions live in
-- "Control.Monad.State.Class".
module Control.Monad.Trans.State.Strict
  ( -- * The State monad
    State
  , runState
  , evalState
  , execState
  , mapState
  , withState
    -- * The StateT transformer
  , StateT (..)
  , evalStateT
  , execStateT
  , mapStateT
  , withStateT
    -- * State operations
  , get
  , put
  , modify
  , modify'
  , gets
  , state
  ) where

import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (MonadTrans (..))
import Data.Functor.Identity (Identity (..))

-- | A state transformer: a function from an initial state to an action in
-- the inner monad yielding a result and a final state.
newtype StateT s m a = StateT { runStateT :: s -> m (a, s) }

-- | A state monad over 'Identity'.
type State s = StateT s Identity

instance Functor m => Functor (StateT s m) where
  fmap f (StateT g) = StateT (\s -> fmap (\(a, s') -> (f a, s')) (g s))

instance Monad m => Applicative (StateT s m) where
  pure a = StateT (\s -> return (a, s))
  StateT mf <*> StateT mx = StateT (\s -> mf s >>= \(f, s1) -> mx s1 >>= \(x, s2) -> return (f x, s2))

instance Monad m => Monad (StateT s m) where
  StateT m >>= k = StateT (\s -> m s >>= \(a, s1) -> runStateT (k a) s1)

instance MonadTrans (StateT s) where
  lift m = StateT (\s -> m >>= \a -> return (a, s))

instance MonadIO m => MonadIO (StateT s m) where
  liftIO io = lift (liftIO io)

-- | Run a 'State' computation from an initial state.
runState :: State s a -> s -> (a, s)
runState m s = runIdentity (runStateT m s)

-- | The result of a 'State' computation.
evalState :: State s a -> s -> a
evalState m s = fst (runState m s)

-- | The final state of a 'State' computation.
execState :: State s a -> s -> s
execState m s = snd (runState m s)

-- | Transform the result and final state of a 'State' computation.
mapState :: ((a, s) -> (b, s)) -> State s a -> State s b
mapState f = mapStateT (\m -> Identity (f (runIdentity m)))

-- | Modify the initial state before running.
withState :: (s -> s) -> State s a -> State s a
withState = withStateT

-- | The result of a 'StateT' computation.
evalStateT :: Monad m => StateT s m a -> s -> m a
evalStateT m s = runStateT m s >>= \(a, _) -> return a

-- | The final state of a 'StateT' computation.
execStateT :: Monad m => StateT s m a -> s -> m s
execStateT m s = runStateT m s >>= \(_, s') -> return s'

-- | Transform the inner action.
mapStateT :: (m (a, s) -> n (b, s)) -> StateT s m a -> StateT s n b
mapStateT f (StateT g) = StateT (\s -> f (g s))

-- | Modify the initial state before running.
withStateT :: (s -> s) -> StateT s m a -> StateT s m a
withStateT f (StateT g) = StateT (\s -> g (f s))

-- | The current state.
get :: Monad m => StateT s m s
get = StateT (\s -> return (s, s))

-- | Replace the state.
put :: Monad m => s -> StateT s m ()
put s = StateT (\_ -> return ((), s))

-- | Apply a function to the state.
modify :: Monad m => (s -> s) -> StateT s m ()
modify f = StateT (\s -> return ((), f s))

-- | Apply a function to the state, forcing the new state.
modify' :: Monad m => (s -> s) -> StateT s m ()
modify' f = StateT (\s -> let s' = f s in s' `seq` return ((), s'))

-- | A projection of the current state.
gets :: Monad m => (s -> a) -> StateT s m a
gets f = StateT (\s -> return (f s, s))

-- | Embed a state transition.
state :: Monad m => (s -> (a, s)) -> StateT s m a
state f = StateT (\s -> return (f s))
