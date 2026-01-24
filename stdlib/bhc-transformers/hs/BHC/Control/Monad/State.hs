-- |
-- Module      : BHC.Control.Monad.State
-- Description : State monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

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
type State s = StateT s Identity

-- | Run a State.
runState :: State s a -> s -> (a, s)
runState m s = runIdentity (runStateT m s)

-- | Run and return only the result.
evalState :: State s a -> s -> a
evalState m s = fst (runState m s)

-- | Run and return only the final state.
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

-- | Fetch the current state.
get :: Monad m => StateT s m s
get = StateT $ \s -> return (s, s)

-- | Set the state.
put :: Monad m => s -> StateT s m ()
put s = StateT $ \_ -> return ((), s)

-- | Modify the state.
modify :: Monad m => (s -> s) -> StateT s m ()
modify f = StateT $ \s -> return ((), f s)

-- | Strict modify.
modify' :: Monad m => (s -> s) -> StateT s m ()
modify' f = StateT $ \s -> let s' = f s in s' `seq` return ((), s')

-- | Get a function of the state.
gets :: Monad m => (s -> a) -> StateT s m a
gets f = StateT $ \s -> return (f s, s)

-- | Create state from a function.
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
