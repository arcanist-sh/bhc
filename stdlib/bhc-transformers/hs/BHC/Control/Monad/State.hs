-- |
-- Module      : BHC.Control.Monad.State
-- Description : State monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The State monad for computations with mutable state.
--
-- = Overview
--
-- The State monad represents computations that carry a modifiable state.
-- This is useful for:
--
-- * Accumulating results during a traversal
-- * Unique ID generation
-- * Stateful parsing
-- * Simulating mutable variables in pure code
--
-- = Usage
--
-- @
-- import BHC.Control.Monad.State
--
-- -- Generate unique IDs
-- fresh :: State Int Int
-- fresh = do
--     n <- get
--     put (n + 1)
--     return n
--
-- -- Run with initial state
-- threeIds :: [Int]
-- threeIds = evalState (replicateM 3 fresh) 0
-- -- Result: [0, 1, 2]
-- @
--
-- = Strict vs Lazy
--
-- This module provides a strict 'State' monad. Use 'modify'' for strict
-- state updates to avoid space leaks from accumulated thunks.

module BHC.Control.Monad.State (
    -- * The State monad
    State,
    runState,
    evalState,
    execState,
    mapState,
    withState,
    
    -- * The StateT monad transformer
    StateT(..),
    runStateT,
    evalStateT,
    execStateT,
    mapStateT,
    withStateT,
    
    -- * State operations
    get,
    put,
    modify,
    modify',
    gets,
    state,
    
    -- * MonadState class
    MonadState(..),
) where

import BHC.Prelude
import BHC.Control.Monad.Trans
import BHC.Control.Monad.Identity (Identity(..))

-- | The state monad.
--
-- @State s a@ is a computation that threads state of type @s@ and
-- produces a value of type @a@.
type State s = StateT s Identity

-- | /O(1)/. Run a 'State' computation with an initial state.
--
-- Returns both the result and the final state.
--
-- ==== __Examples__
--
-- >>> runState (modify (+1) >> get) 0
-- (1, 1)
runState :: State s a -> s -> (a, s)
runState m s = runIdentity (runStateT m s)

-- | /O(1)/. Run a 'State' computation, returning only the result.
--
-- ==== __Examples__
--
-- >>> evalState (replicateM 3 (state (\n -> (n, n+1)))) 0
-- [0, 1, 2]
evalState :: State s a -> s -> a
evalState m s = fst (runState m s)

-- | /O(1)/. Run a 'State' computation, returning only the final state.
--
-- ==== __Examples__
--
-- >>> execState (modify (+1) >> modify (*2)) 5
-- 12
execState :: State s a -> s -> s
execState m s = snd (runState m s)

-- | Map both the return value and state.
mapState :: ((a, s) -> (b, s)) -> State s a -> State s b
mapState f = mapStateT (Identity . f . runIdentity)

-- | Transform the state.
withState :: (s -> s) -> State s a -> State s a
withState = withStateT

-- | The state monad transformer.
newtype StateT s m a = StateT { runStateT :: s -> m (a, s) }

instance Functor m => Functor (StateT s m) where
    fmap f (StateT g) = StateT $ \s -> fmap (\(a, s') -> (f a, s')) (g s)

instance Monad m => Applicative (StateT s m) where
    pure a = StateT $ \s -> return (a, s)
    StateT mf <*> StateT mx = StateT $ \s -> do
        (f, s') <- mf s
        (x, s'') <- mx s'
        return (f x, s'')

instance Monad m => Monad (StateT s m) where
    StateT m >>= k = StateT $ \s -> do
        (a, s') <- m s
        runStateT (k a) s'

instance MonadTrans (StateT s) where
    lift m = StateT $ \s -> do
        a <- m
        return (a, s)

instance MonadIO m => MonadIO (StateT s m) where
    liftIO = lift . liftIO

-- | Run and return only the result.
evalStateT :: Monad m => StateT s m a -> s -> m a
evalStateT m s = fmap fst (runStateT m s)

-- | Run and return only the final state.
execStateT :: Monad m => StateT s m a -> s -> m s
execStateT m s = fmap snd (runStateT m s)

-- | Map the inner computation.
mapStateT :: (m (a, s) -> n (b, s)) -> StateT s m a -> StateT s n b
mapStateT f (StateT g) = StateT (f . g)

-- | Transform the state.
withStateT :: (s -> s) -> StateT s m a -> StateT s m a
withStateT f (StateT g) = StateT (g . f)

-- | /O(1)/. Retrieve the current state.
--
-- ==== __Examples__
--
-- >>> evalState get 42
-- 42
get :: Monad m => StateT s m s
get = StateT $ \s -> return (s, s)

-- | /O(1)/. Replace the state with a new value.
--
-- ==== __Examples__
--
-- >>> execState (put 99) 0
-- 99
put :: Monad m => s -> StateT s m ()
put s = StateT $ \_ -> return ((), s)

-- | /O(1)/. Modify the state by applying a function.
--
-- __Note__: This is lazy in the new state. Use 'modify'' for strict updates.
--
-- ==== __Examples__
--
-- >>> execState (modify (*2)) 5
-- 10
modify :: Monad m => (s -> s) -> StateT s m ()
modify f = StateT $ \s -> return ((), f s)

-- | /O(1)/. Strictly modify the state.
--
-- Forces the new state before storing it, preventing space leaks
-- from accumulated thunks. Prefer this over 'modify' for numeric
-- counters and similar use cases.
--
-- ==== __Examples__
--
-- @
-- -- Avoid stack overflow from lazy accumulation
-- countTo :: Int -> Int
-- countTo n = execState (replicateM_ n (modify' (+1))) 0
-- @
modify' :: Monad m => (s -> s) -> StateT s m ()
modify' f = StateT $ \s -> let s' = f s in s' `seq` return ((), s')

-- | /O(1)/. Retrieve a function of the current state.
--
-- @gets f@ is equivalent to @fmap f get@, but more efficient.
--
-- ==== __Examples__
--
-- >>> evalState (gets length) [1,2,3]
-- 3
gets :: Monad m => (s -> a) -> StateT s m a
gets f = StateT $ \s -> return (f s, s)

-- | /O(1)/. Embed a state action into the monad.
--
-- ==== __Examples__
--
-- @
-- fresh :: State Int Int
-- fresh = state (\\n -> (n, n + 1))
-- @
state :: Monad m => (s -> (a, s)) -> StateT s m a
state f = StateT (return . f)

-- | Class for monads with state.
class Monad m => MonadState s m | m -> s where
    getS :: m s
    putS :: s -> m ()
    stateS :: (s -> (a, s)) -> m a
    
    stateS f = do
        s <- getS
        let (a, s') = f s
        putS s'
        return a

instance Monad m => MonadState s (StateT s m) where
    getS = get
    putS = put
    stateS = state
