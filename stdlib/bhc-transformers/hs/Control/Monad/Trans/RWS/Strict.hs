-- |
-- Module      : Control.Monad.Trans.RWS.Strict
-- Description : The reader-writer-state monad transformer
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005). A computation
-- reads an environment @r@, accumulates output @w@ and threads a state @s@.
module Control.Monad.Trans.RWS.Strict
  ( -- * The RWS monad
    RWS
  , rws
  , runRWS
  , evalRWS
  , execRWS
    -- * The RWST transformer
  , RWST (..)
  , evalRWST
  , execRWST
    -- * Reader operations
  , ask
  , local
  , asks
  , reader
    -- * Writer operations
  , tell
  , writer
  , listen
  , pass
    -- * State operations
  , get
  , put
  , modify
  , gets
  , state
  ) where

import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (MonadTrans (..))
import Data.Functor.Identity (Identity (..))

-- | A reader-writer-state computation over the inner monad @m@.
newtype RWST r w s m a = RWST { runRWST :: r -> s -> m (a, s, w) }

-- | 'RWST' over 'Identity'.
type RWS r w s = RWST r w s Identity

instance Functor m => Functor (RWST r w s m) where
  fmap f m = RWST (\r s -> fmap (\(a, s', w) -> (f a, s', w)) (runRWST m r s))

instance (Monoid w, Monad m) => Applicative (RWST r w s m) where
  pure a = RWST (\_ s -> return (a, s, mempty))
  mf <*> mx = RWST (\r s ->
    runRWST mf r s >>= \(f, s1, w1) ->
    runRWST mx r s1 >>= \(x, s2, w2) ->
    return (f x, s2, w1 <> w2))

instance (Monoid w, Monad m) => Monad (RWST r w s m) where
  m >>= k = RWST (\r s ->
    runRWST m r s >>= \(a, s1, w1) ->
    runRWST (k a) r s1 >>= \(b, s2, w2) ->
    return (b, s2, w1 <> w2))

instance Monoid w => MonadTrans (RWST r w s) where
  lift m = RWST (\_ s -> m >>= \a -> return (a, s, mempty))

instance (Monoid w, MonadIO m) => MonadIO (RWST r w s m) where
  liftIO io = lift (liftIO io)

-- | Build an 'RWS' computation from its function.
rws :: Monoid w => (r -> s -> (a, s, w)) -> RWS r w s a
rws f = RWST (\r s -> Identity (f r s))

-- | Run an 'RWS' computation.
runRWS :: RWS r w s a -> r -> s -> (a, s, w)
runRWS m r s = runIdentity (runRWST m r s)

-- | Run an 'RWS' computation, keeping its result and output.
evalRWS :: RWS r w s a -> r -> s -> (a, w)
evalRWS m r s = case runRWS m r s of
  (a, _, w) -> (a, w)

-- | Run an 'RWS' computation, keeping its final state and output.
execRWS :: RWS r w s a -> r -> s -> (s, w)
execRWS m r s = case runRWS m r s of
  (_, s', w) -> (s', w)

-- | Run an 'RWST' computation, keeping its result and output.
evalRWST :: Monad m => RWST r w s m a -> r -> s -> m (a, w)
evalRWST m r s = runRWST m r s >>= \(a, _, w) -> return (a, w)

-- | Run an 'RWST' computation, keeping its final state and output.
execRWST :: Monad m => RWST r w s m a -> r -> s -> m (s, w)
execRWST m r s = runRWST m r s >>= \(_, s', w) -> return (s', w)

-- | The environment.
ask :: (Monoid w, Monad m) => RWST r w s m r
ask = RWST (\r s -> return (r, s, mempty))

-- | Run a computation in a modified environment.
local :: (r -> r) -> RWST r w s m a -> RWST r w s m a
local f m = RWST (\r s -> runRWST m (f r) s)

-- | A projection of the environment.
asks :: (Monoid w, Monad m) => (r -> a) -> RWST r w s m a
asks f = RWST (\r s -> return (f r, s, mempty))

-- | A projection of the environment ('asks').
reader :: (Monoid w, Monad m) => (r -> a) -> RWST r w s m a
reader = asks

-- | Append to the output.
tell :: Monad m => w -> RWST r w s m ()
tell w = RWST (\_ s -> return ((), s, w))

-- | A result together with output.
writer :: Monad m => (a, w) -> RWST r w s m a
writer (a, w) = RWST (\_ s -> return (a, s, w))

-- | Also return the output a computation produced.
listen :: Monad m => RWST r w s m a -> RWST r w s m (a, w)
listen m = RWST (\r s -> runRWST m r s >>= \(a, s', w) -> return ((a, w), s', w))

-- | Apply the function a computation returns to its output.
pass :: Monad m => RWST r w s m (a, w -> w) -> RWST r w s m a
pass m = RWST (\r s -> runRWST m r s >>= \((a, f), s', w) -> return (a, s', f w))

-- | The state.
get :: (Monoid w, Monad m) => RWST r w s m s
get = RWST (\_ s -> return (s, s, mempty))

-- | Replace the state.
put :: (Monoid w, Monad m) => s -> RWST r w s m ()
put s = RWST (\_ _ -> return ((), s, mempty))

-- | Apply a function to the state.
modify :: (Monoid w, Monad m) => (s -> s) -> RWST r w s m ()
modify f = RWST (\_ s -> return ((), f s, mempty))

-- | A projection of the state.
gets :: (Monoid w, Monad m) => (s -> a) -> RWST r w s m a
gets f = RWST (\_ s -> return (f s, s, mempty))

-- | A state transition.
state :: (Monoid w, Monad m) => (s -> (a, s)) -> RWST r w s m a
state f = RWST (\_ s -> case f s of
  (a, s') -> return (a, s', mempty))
