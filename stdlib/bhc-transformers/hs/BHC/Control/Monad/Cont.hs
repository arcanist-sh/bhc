-- |
-- Module      : BHC.Control.Monad.Cont
-- Description : Continuation monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The ContT monad transformer provides continuation-passing style.
--
-- Continuations are a powerful control flow mechanism that can be used
-- to implement exceptions, coroutines, generators, and more.
--
-- = Usage
--
-- @
-- import BHC.Control.Monad.Cont
--
-- -- Early return using callCC
-- divSafe :: Int -> Int -> Cont String Int
-- divSafe x y = callCC $ \\k -> do
--     when (y == 0) $ k 0  -- Early return
--     return (x \`div\` y)
-- @

module BHC.Control.Monad.Cont
    ( -- * The Cont monad
      Cont
    , cont
    , runCont
    , evalCont
    , mapCont
    , withCont

      -- * The ContT monad transformer
    , ContT(..)
    , runContT
    , evalContT
    , mapContT
    , withContT

      -- * Continuation operations
    , callCC
    , reset
    , shift

      -- * MonadCont class
    , MonadCont(..)
    ) where

import BHC.Prelude
import BHC.Control.Monad.Trans
import BHC.Control.Monad.Identity

-- ============================================================
-- Cont monad (non-transformer)
-- ============================================================

-- | The continuation monad.
--
-- Computation type: Continuation-passing style (CPS) computations.
-- Binding strategy: Passes the continuation to the next computation.
-- Useful for: Complex control flow (early exit, coroutines, etc.)
type Cont r = ContT r Identity

-- | Construct a continuation.
cont :: ((a -> r) -> r) -> Cont r a
cont f = ContT $ \k -> Identity (f (runIdentity . k))
{-# INLINE cont #-}

-- | Run a continuation with a final continuation.
runCont :: Cont r a -> (a -> r) -> r
runCont m k = runIdentity (runContT m (Identity . k))
{-# INLINE runCont #-}

-- | Evaluate a continuation using 'id' as the final continuation.
evalCont :: Cont r r -> r
evalCont m = runCont m id
{-# INLINE evalCont #-}

-- | Map the result of a continuation.
mapCont :: (r -> r) -> Cont r a -> Cont r a
mapCont f = mapContT (Identity . f . runIdentity)
{-# INLINE mapCont #-}

-- | Transform the continuation.
withCont :: ((b -> r) -> a -> r) -> Cont r a -> Cont r b
withCont f = withContT ((Identity .) . f . (runIdentity .))
{-# INLINE withCont #-}

-- ============================================================
-- ContT monad transformer
-- ============================================================

-- | The continuation monad transformer.
--
-- The ContT monad transformer adds continuation-passing style
-- to an underlying monad.
newtype ContT r m a = ContT { runContT :: (a -> m r) -> m r }

instance Functor (ContT r m) where
    fmap f (ContT c) = ContT $ \k -> c (k . f)
    {-# INLINE fmap #-}

instance Applicative (ContT r m) where
    pure a = ContT $ \k -> k a
    {-# INLINE pure #-}

    ContT cf <*> ContT cx = ContT $ \k -> cf $ \f -> cx $ \x -> k (f x)
    {-# INLINE (<*>) #-}

instance Monad (ContT r m) where
    ContT c >>= f = ContT $ \k -> c $ \a -> runContT (f a) k
    {-# INLINE (>>=) #-}

instance MonadTrans (ContT r) where
    lift m = ContT $ \k -> m >>= k
    {-# INLINE lift #-}

instance MonadIO m => MonadIO (ContT r m) where
    liftIO = lift . liftIO
    {-# INLINE liftIO #-}

-- | Evaluate a continuation using 'return' as the final continuation.
evalContT :: Monad m => ContT r m r -> m r
evalContT m = runContT m return
{-# INLINE evalContT #-}

-- | Map the result of a continuation.
mapContT :: (m r -> m r) -> ContT r m a -> ContT r m a
mapContT f (ContT c) = ContT $ \k -> f (c k)
{-# INLINE mapContT #-}

-- | Transform the continuation.
withContT :: ((b -> m r) -> a -> m r) -> ContT r m a -> ContT r m b
withContT f (ContT c) = ContT $ \k -> c (f k)
{-# INLINE withContT #-}

-- ============================================================
-- Continuation operations
-- ============================================================

-- | Call with current continuation.
--
-- @callCC@ captures the current continuation and passes it to
-- the given function. The continuation can be called to immediately
-- return from the callCC block with a value.
--
-- ==== __Examples__
--
-- >>> evalCont $ callCC $ \k -> do { k 1; return 2 }
-- 1
--
-- >>> evalCont $ callCC $ \k -> do { return 2 }
-- 2
callCC :: ((a -> ContT r m b) -> ContT r m a) -> ContT r m a
callCC f = ContT $ \k -> runContT (f (\a -> ContT $ \_ -> k a)) k
{-# INLINE callCC #-}

-- | Delimit the continuation.
--
-- The continuation captured by 'shift' is delimited by the
-- enclosing 'reset'.
--
-- ==== __Examples__
--
-- >>> evalCont $ reset $ do { x <- shift $ \k -> k 1; return (x + 1) }
-- 2
reset :: Monad m => ContT r m r -> ContT r' m r
reset m = ContT $ \k -> evalContT m >>= k
{-# INLINE reset #-}

-- | Capture the delimited continuation.
--
-- 'shift' captures the continuation up to the enclosing 'reset'
-- and passes it to the given function.
--
-- ==== __Examples__
--
-- >>> evalCont $ reset $ do { shift $ \k -> do { x <- k 1; k (x + 1) }; return 0 }
-- 1
shift :: Monad m => ((a -> m r) -> ContT r m r) -> ContT r m a
shift f = ContT $ \k -> evalContT (f k)
{-# INLINE shift #-}

-- ============================================================
-- MonadCont class
-- ============================================================

-- | Class of monads that support callCC.
class Monad m => MonadCont m where
    -- | Call with current continuation.
    callCCM :: ((a -> m b) -> m a) -> m a

instance MonadCont (ContT r m) where
    callCCM = callCC
    {-# INLINE callCCM #-}

instance MonadCont (Cont r) where
    callCCM f = cont $ \k -> runCont (f (\a -> cont $ \_ -> k a)) k
    {-# INLINE callCCM #-}
