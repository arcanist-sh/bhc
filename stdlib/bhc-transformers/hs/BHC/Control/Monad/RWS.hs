-- |
-- Module      : BHC.Control.Monad.RWS
-- Description : Combined Reader-Writer-State monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The RWST monad transformer combines 'ReaderT', 'WriterT', and 'StateT'
-- into a single transformer for efficiency.
--
-- = Usage
--
-- @
-- import BHC.Control.Monad.RWS
--
-- type App = RWST Config Log AppState IO
--
-- runApp :: App a -> Config -> AppState -> IO (a, AppState, Log)
-- runApp = runRWST
-- @

module BHC.Control.Monad.RWS
    ( -- * The RWS monad
      RWS
    , rws
    , runRWS
    , evalRWS
    , execRWS
    , mapRWS
    , withRWS

      -- * The RWST monad transformer
    , RWST(..)
    , runRWST
    , evalRWST
    , execRWST
    , mapRWST
    , withRWST

      -- * Reader operations
    , ask
    , local
    , asks
    , reader

      -- * Writer operations
    , tell
    , listen
    , listens
    , pass
    , censor
    , writer

      -- * State operations
    , get
    , put
    , modify
    , modify'
    , gets
    , state
    ) where

import BHC.Prelude
import BHC.Control.Monad.Trans
import BHC.Control.Monad.Identity

-- ============================================================
-- RWS monad (non-transformer)
-- ============================================================

-- | A monad containing an environment of type @r@, output of type @w@,
-- and an updatable state of type @s@.
type RWS r w s = RWST r w s Identity

-- | Construct an RWS from a function.
rws :: (r -> s -> (a, s, w)) -> RWS r w s a
rws f = RWST $ \r s -> Identity (f r s)
{-# INLINE rws #-}

-- | Run an RWS.
runRWS :: RWS r w s a -> r -> s -> (a, s, w)
runRWS m r s = runIdentity (runRWST m r s)
{-# INLINE runRWS #-}

-- | Evaluate an RWS, returning only the result.
evalRWS :: RWS r w s a -> r -> s -> (a, w)
evalRWS m r s = let (a, _, w) = runRWS m r s in (a, w)
{-# INLINE evalRWS #-}

-- | Execute an RWS, returning only the final state and output.
execRWS :: RWS r w s a -> r -> s -> (s, w)
execRWS m r s = let (_, s', w) = runRWS m r s in (s', w)
{-# INLINE execRWS #-}

-- | Map the return value, final state, and output of an RWS.
mapRWS :: ((a, s, w) -> (b, s, w')) -> RWS r w s a -> RWS r w' s b
mapRWS f = mapRWST (Identity . f . runIdentity)
{-# INLINE mapRWS #-}

-- | Execute an RWS with a modified environment.
withRWS :: (r' -> s -> (r, s)) -> RWS r w s a -> RWS r' w s a
withRWS = withRWST
{-# INLINE withRWS #-}

-- ============================================================
-- RWST monad transformer
-- ============================================================

-- | A monad transformer containing an environment of type @r@,
-- output of type @w@, and an updatable state of type @s@.
--
-- This is more efficient than stacking 'ReaderT', 'WriterT', and 'StateT'
-- because it only wraps the underlying monad once.
newtype RWST r w s m a = RWST { runRWST :: r -> s -> m (a, s, w) }

instance Functor m => Functor (RWST r w s m) where
    fmap f (RWST m) = RWST $ \r s -> fmap (\(a, s', w) -> (f a, s', w)) (m r s)
    {-# INLINE fmap #-}

instance (Monoid w, Monad m) => Applicative (RWST r w s m) where
    pure a = RWST $ \_ s -> return (a, s, mempty)
    {-# INLINE pure #-}

    RWST mf <*> RWST mx = RWST $ \r s -> do
        (f, s', w1) <- mf r s
        (x, s'', w2) <- mx r s'
        return (f x, s'', w1 <> w2)
    {-# INLINE (<*>) #-}

instance (Monoid w, Monad m) => Monad (RWST r w s m) where
    RWST m >>= k = RWST $ \r s -> do
        (a, s', w1) <- m r s
        (b, s'', w2) <- runRWST (k a) r s'
        return (b, s'', w1 <> w2)
    {-# INLINE (>>=) #-}

instance Monoid w => MonadTrans (RWST r w s) where
    lift m = RWST $ \_ s -> do
        a <- m
        return (a, s, mempty)
    {-# INLINE lift #-}

instance (Monoid w, MonadIO m) => MonadIO (RWST r w s m) where
    liftIO = lift . liftIO
    {-# INLINE liftIO #-}

-- | Evaluate an RWST, returning only the result and output.
evalRWST :: Monad m => RWST r w s m a -> r -> s -> m (a, w)
evalRWST m r s = do
    (a, _, w) <- runRWST m r s
    return (a, w)
{-# INLINE evalRWST #-}

-- | Execute an RWST, returning only the final state and output.
execRWST :: Monad m => RWST r w s m a -> r -> s -> m (s, w)
execRWST m r s = do
    (_, s', w) <- runRWST m r s
    return (s', w)
{-# INLINE execRWST #-}

-- | Map the inner computation.
mapRWST :: (m (a, s, w) -> n (b, s, w')) -> RWST r w s m a -> RWST r w' s n b
mapRWST f (RWST m) = RWST $ \r s -> f (m r s)
{-# INLINE mapRWST #-}

-- | Execute with a modified environment and state.
withRWST :: (r' -> s -> (r, s)) -> RWST r w s m a -> RWST r' w s m a
withRWST f (RWST m) = RWST $ \r s -> let (r', s') = f r s in m r' s'
{-# INLINE withRWST #-}

-- ============================================================
-- Reader operations
-- ============================================================

-- | Fetch the environment.
ask :: Monad m => RWST r w s m r
ask = RWST $ \r s -> return (r, s, mempty)
{-# INLINE ask #-}

-- | Execute with a modified environment.
local :: (r -> r) -> RWST r w s m a -> RWST r w s m a
local f (RWST m) = RWST $ \r s -> m (f r) s
{-# INLINE local #-}

-- | Fetch a function of the environment.
asks :: Monad m => (r -> a) -> RWST r w s m a
asks f = RWST $ \r s -> return (f r, s, mempty)
{-# INLINE asks #-}

-- | Create a reader computation.
reader :: Monad m => (r -> a) -> RWST r w s m a
reader = asks
{-# INLINE reader #-}

-- ============================================================
-- Writer operations
-- ============================================================

-- | Append to the output.
tell :: Monad m => w -> RWST r w s m ()
tell w = RWST $ \_ s -> return ((), s, w)
{-# INLINE tell #-}

-- | Execute and collect the output.
listen :: Monad m => RWST r w s m a -> RWST r w s m (a, w)
listen (RWST m) = RWST $ \r s -> do
    (a, s', w) <- m r s
    return ((a, w), s', w)
{-# INLINE listen #-}

-- | Execute and apply a function to the output.
listens :: Monad m => (w -> b) -> RWST r w s m a -> RWST r w s m (a, b)
listens f m = do
    (a, w) <- listen m
    return (a, f w)
{-# INLINE listens #-}

-- | Execute with a function that can modify output.
pass :: Monad m => RWST r w s m (a, w -> w) -> RWST r w s m a
pass (RWST m) = RWST $ \r s -> do
    ((a, f), s', w) <- m r s
    return (a, s', f w)
{-# INLINE pass #-}

-- | Apply a function to the output.
censor :: Monad m => (w -> w) -> RWST r w s m a -> RWST r w s m a
censor f m = pass $ do
    a <- m
    return (a, f)
{-# INLINE censor #-}

-- | Create a writer computation.
writer :: Monad m => (a, w) -> RWST r w s m a
writer (a, w) = RWST $ \_ s -> return (a, s, w)
{-# INLINE writer #-}

-- ============================================================
-- State operations
-- ============================================================

-- | Fetch the current state.
get :: Monad m => RWST r w s m s
get = RWST $ \_ s -> return (s, s, mempty)
{-# INLINE get #-}

-- | Set the state.
put :: Monad m => s -> RWST r w s m ()
put s = RWST $ \_ _ -> return ((), s, mempty)
{-# INLINE put #-}

-- | Modify the state.
modify :: Monad m => (s -> s) -> RWST r w s m ()
modify f = RWST $ \_ s -> return ((), f s, mempty)
{-# INLINE modify #-}

-- | Strict modify.
modify' :: Monad m => (s -> s) -> RWST r w s m ()
modify' f = RWST $ \_ s -> let s' = f s in s' `seq` return ((), s', mempty)
{-# INLINE modify' #-}

-- | Get a function of the state.
gets :: Monad m => (s -> a) -> RWST r w s m a
gets f = RWST $ \_ s -> return (f s, s, mempty)
{-# INLINE gets #-}

-- | Create a state computation.
state :: Monad m => (s -> (a, s)) -> RWST r w s m a
state f = RWST $ \_ s -> let (a, s') = f s in return (a, s', mempty)
{-# INLINE state #-}
