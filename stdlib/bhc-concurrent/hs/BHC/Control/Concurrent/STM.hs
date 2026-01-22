-- |
-- Module      : BHC.Control.Concurrent.STM
-- Description : Software Transactional Memory
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Software Transactional Memory for safe concurrent access
-- to shared state.

module BHC.Control.Concurrent.STM (
    -- * STM monad
    STM,
    atomically,
    
    -- * TVars
    TVar,
    newTVar,
    newTVarIO,
    readTVar,
    writeTVar,
    modifyTVar,
    modifyTVar',
    swapTVar,
    
    -- * Retry and choice
    retry,
    orElse,
    check,
    
    -- * TMVar
    TMVar,
    newTMVar,
    newTMVarIO,
    newEmptyTMVar,
    newEmptyTMVarIO,
    takeTMVar,
    putTMVar,
    readTMVar,
    tryTakeTMVar,
    tryPutTMVar,
    isEmptyTMVar,
    swapTMVar,
    
    -- * TChan
    TChan,
    newTChan,
    newTChanIO,
    readTChan,
    writeTChan,
    dupTChan,
    isEmptyTChan,
    
    -- * Utilities
    throwSTM,
    catchSTM,
) where

import BHC.Prelude

-- | The STM monad for transactional operations.
data STM a

instance Functor STM where
    fmap = liftM

instance Applicative STM where
    pure = return
    (<*>) = ap

instance Monad STM where
    return = returnSTM
    (>>=) = bindSTM

-- | Execute an STM transaction atomically.
foreign import ccall "bhc_atomically"
    atomically :: STM a -> IO a

foreign import ccall "bhc_stm_return"
    returnSTM :: a -> STM a

foreign import ccall "bhc_stm_bind"
    bindSTM :: STM a -> (a -> STM b) -> STM b

-- | A transactional variable.
data TVar a

-- | Create a new TVar.
foreign import ccall "bhc_new_tvar"
    newTVar :: a -> STM (TVar a)

-- | Create a new TVar in IO.
foreign import ccall "bhc_new_tvar_io"
    newTVarIO :: a -> IO (TVar a)

-- | Read the value of a TVar.
foreign import ccall "bhc_read_tvar"
    readTVar :: TVar a -> STM a

-- | Write a value to a TVar.
foreign import ccall "bhc_write_tvar"
    writeTVar :: TVar a -> a -> STM ()

-- | Modify a TVar.
modifyTVar :: TVar a -> (a -> a) -> STM ()
modifyTVar v f = readTVar v >>= writeTVar v . f

-- | Strict modify.
modifyTVar' :: TVar a -> (a -> a) -> STM ()
modifyTVar' v f = do
    x <- readTVar v
    writeTVar v $! f x

-- | Swap the value of a TVar.
swapTVar :: TVar a -> a -> STM a
swapTVar v new = do
    old <- readTVar v
    writeTVar v new
    return old

-- | Retry the transaction.
foreign import ccall "bhc_retry"
    retry :: STM a

-- | Try the first action, or the second if it retries.
foreign import ccall "bhc_or_else"
    orElse :: STM a -> STM a -> STM a

-- | Retry if the condition is false.
check :: Bool -> STM ()
check True  = return ()
check False = retry

-- | A transactional MVar.
data TMVar a

foreign import ccall "bhc_new_tmvar"
    newTMVar :: a -> STM (TMVar a)

foreign import ccall "bhc_new_tmvar_io"
    newTMVarIO :: a -> IO (TMVar a)

foreign import ccall "bhc_new_empty_tmvar"
    newEmptyTMVar :: STM (TMVar a)

foreign import ccall "bhc_new_empty_tmvar_io"
    newEmptyTMVarIO :: IO (TMVar a)

foreign import ccall "bhc_take_tmvar"
    takeTMVar :: TMVar a -> STM a

foreign import ccall "bhc_put_tmvar"
    putTMVar :: TMVar a -> a -> STM ()

readTMVar :: TMVar a -> STM a
readTMVar m = do
    x <- takeTMVar m
    putTMVar m x
    return x

tryTakeTMVar :: TMVar a -> STM (Maybe a)
tryTakeTMVar m = (Just <$> takeTMVar m) `orElse` return Nothing

tryPutTMVar :: TMVar a -> a -> STM Bool
tryPutTMVar m x = (putTMVar m x >> return True) `orElse` return False

foreign import ccall "bhc_is_empty_tmvar"
    isEmptyTMVar :: TMVar a -> STM Bool

swapTMVar :: TMVar a -> a -> STM a
swapTMVar m new = do
    old <- takeTMVar m
    putTMVar m new
    return old

-- | A transactional channel.
data TChan a

foreign import ccall "bhc_new_tchan"
    newTChan :: STM (TChan a)

foreign import ccall "bhc_new_tchan_io"
    newTChanIO :: IO (TChan a)

foreign import ccall "bhc_read_tchan"
    readTChan :: TChan a -> STM a

foreign import ccall "bhc_write_tchan"
    writeTChan :: TChan a -> a -> STM ()

foreign import ccall "bhc_dup_tchan"
    dupTChan :: TChan a -> STM (TChan a)

foreign import ccall "bhc_is_empty_tchan"
    isEmptyTChan :: TChan a -> STM Bool

-- | Throw an exception in STM.
foreign import ccall "bhc_throw_stm"
    throwSTM :: Exception e => e -> STM a

-- | Catch an exception in STM.
foreign import ccall "bhc_catch_stm"
    catchSTM :: Exception e => STM a -> (e -> STM a) -> STM a
