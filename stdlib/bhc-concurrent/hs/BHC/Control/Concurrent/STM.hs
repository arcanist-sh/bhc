-- |
-- Module      : BHC.Control.Concurrent.STM
-- Description : Software Transactional Memory
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Software Transactional Memory for safe concurrent access
-- to shared state.
--
-- = Overview
--
-- STM provides composable, deadlock-free concurrent programming.
-- Transactions appear to execute atomically, even when running
-- concurrently. If a transaction conflicts, it automatically retries.
--
-- = Quick Start
--
-- @
-- import BHC.Control.Concurrent.STM
--
-- -- Thread-safe counter
-- incrementCounter :: TVar Int -> IO ()
-- incrementCounter counter = atomically $ do
--     n <- readTVar counter
--     writeTVar counter (n + 1)
--
-- -- Composable transactions
-- transfer :: TVar Int -> TVar Int -> Int -> STM ()
-- transfer from to amount = do
--     balance <- readTVar from
--     check (balance >= amount)  -- Retry if insufficient funds
--     writeTVar from (balance - amount)
--     modifyTVar' to (+ amount)
-- @
--
-- = Key Concepts
--
-- == Transactions
--
-- All STM operations run inside the 'STM' monad. Transactions are
-- executed atomically using 'atomically'. A transaction either:
--
-- * Commits successfully (all changes visible atomically)
-- * Retries (via 'retry' or conflict detection)
-- * Aborts (via exception)
--
-- == Retry and Choice
--
-- The 'retry' operation blocks until a 'TVar' read by the transaction
-- changes. Use 'orElse' to compose alternatives:
--
-- @
-- -- Try first channel, fall back to second
-- readEither :: TChan a -> TChan a -> STM a
-- readEither c1 c2 = readTChan c1 \`orElse\` readTChan c2
-- @
--
-- = See Also
--
-- * 'BHC.Control.Concurrent.Scope' for structured concurrency
-- * H26-SPEC Section 10 for STM specification

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
--
-- 'STM' actions are composable transactions that execute atomically.
-- They can read and write 'TVar's, 'TMVar's, and 'TChan's. Transactions
-- cannot perform arbitrary IO — only STM-specific operations.
--
-- ==== __Laws__
--
-- STM satisfies the monad laws and additionally:
--
-- * Transactions are serializable (appear to execute one at a time)
-- * Conflicting transactions automatically retry
-- * 'retry' blocks until a read variable changes
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
--
-- The transaction either commits (all changes visible at once) or
-- retries (no changes visible). Other threads see either the old
-- state or the new state, never an intermediate state.
--
-- ==== __Examples__
--
-- >>> counter <- newTVarIO 0
-- >>> atomically $ modifyTVar' counter (+1)
-- >>> atomically $ readTVar counter
-- 1
--
-- ==== __Retry Behavior__
--
-- If the transaction calls 'retry' (directly or via 'check'), it
-- blocks until at least one 'TVar' it read has changed, then re-runs.
--
-- ==== __Exception Behavior__
--
-- If the transaction throws (via 'throwSTM'), no changes are committed
-- and the exception propagates to the caller.
--
-- ==== __See Also__
--
-- * 'retry' to block and wait for changes
-- * 'orElse' to try alternatives
foreign import ccall "bhc_atomically"
    atomically :: STM a -> IO a

foreign import ccall "bhc_stm_return"
    returnSTM :: a -> STM a

foreign import ccall "bhc_stm_bind"
    bindSTM :: STM a -> (a -> STM b) -> STM b

-- | A transactional variable holding a value of type @a@.
--
-- 'TVar's are the fundamental building block of STM. They can be
-- read and written within transactions, with automatic conflict
-- detection and retry.
data TVar a

-- | /O(1)/. Create a new 'TVar' within a transaction.
--
-- The initial value is set immediately and visible to subsequent
-- operations in the same transaction.
--
-- ==== __Examples__
--
-- >>> atomically $ do
-- >>>     v <- newTVar 0
-- >>>     modifyTVar' v (+1)
-- >>>     readTVar v
-- 1
foreign import ccall "bhc_new_tvar"
    newTVar :: a -> STM (TVar a)

-- | /O(1)/. Create a new 'TVar' in 'IO'.
--
-- Convenience function for creating 'TVar's outside transactions.
--
-- ==== __Examples__
--
-- >>> counter <- newTVarIO 0
-- >>> atomically $ readTVar counter
-- 0
foreign import ccall "bhc_new_tvar_io"
    newTVarIO :: a -> IO (TVar a)

-- | /O(1)/. Read the current value of a 'TVar'.
--
-- The read is recorded; if another transaction modifies this 'TVar'
-- before commit, the current transaction will retry.
--
-- ==== __Examples__
--
-- >>> v <- newTVarIO 42
-- >>> atomically $ readTVar v
-- 42
foreign import ccall "bhc_read_tvar"
    readTVar :: TVar a -> STM a

-- | /O(1)/. Write a value to a 'TVar'.
--
-- The write is buffered until commit. Other transactions see the
-- new value only after this transaction commits.
--
-- ==== __Examples__
--
-- >>> v <- newTVarIO 0
-- >>> atomically $ writeTVar v 42
-- >>> atomically $ readTVar v
-- 42
foreign import ccall "bhc_write_tvar"
    writeTVar :: TVar a -> a -> STM ()

-- | /O(1)/. Modify the value of a 'TVar'.
--
-- __Note__: This is lazy in the new value. For strict updates that
-- avoid space leaks, use 'modifyTVar''.
--
-- ==== __Examples__
--
-- >>> v <- newTVarIO [1, 2, 3]
-- >>> atomically $ modifyTVar v (4:)
-- >>> atomically $ readTVar v
-- [4, 1, 2, 3]
modifyTVar :: TVar a -> (a -> a) -> STM ()
modifyTVar v f = readTVar v >>= writeTVar v . f

-- | /O(1)/. Strictly modify the value of a 'TVar'.
--
-- Forces the new value before writing, preventing space leaks from
-- accumulated thunks. Prefer this over 'modifyTVar' for numeric
-- counters and similar use cases.
--
-- ==== __Examples__
--
-- >>> counter <- newTVarIO 0
-- >>> atomically $ modifyTVar' counter (+1)
-- >>> atomically $ readTVar counter
-- 1
modifyTVar' :: TVar a -> (a -> a) -> STM ()
modifyTVar' v f = do
    x <- readTVar v
    writeTVar v $! f x

-- | /O(1)/. Swap the value of a 'TVar', returning the old value.
--
-- ==== __Examples__
--
-- >>> v <- newTVarIO "old"
-- >>> atomically $ swapTVar v "new"
-- "old"
-- >>> atomically $ readTVar v
-- "new"
swapTVar :: TVar a -> a -> STM a
swapTVar v new = do
    old <- readTVar v
    writeTVar v new
    return old

-- | Abort and retry the transaction when conditions change.
--
-- Calling 'retry' abandons the current transaction and blocks until
-- at least one 'TVar' read by the transaction has been modified by
-- another thread. Then the transaction restarts from the beginning.
--
-- ==== __Examples__
--
-- @
-- -- Block until the queue is non-empty
-- dequeue :: TVar [a] -> STM a
-- dequeue qVar = do
--     q <- readTVar qVar
--     case q of
--         []     -> retry  -- Block until queue changes
--         (x:xs) -> do
--             writeTVar qVar xs
--             return x
-- @
--
-- ==== __See Also__
--
-- * 'check' for condition-based retry
-- * 'orElse' for trying alternatives
foreign import ccall "bhc_retry"
    retry :: STM a

-- | Try the first action; if it retries, try the second.
--
-- This is the fundamental choice operator for STM. If the first
-- action calls 'retry', its effects are discarded and the second
-- action runs. If the second also retries, the combined action retries.
--
-- ==== __Examples__
--
-- @
-- -- Read from either channel (whichever has data first)
-- readEither :: TChan a -> TChan a -> STM a
-- readEither c1 c2 = readTChan c1 \`orElse\` readTChan c2
-- @
--
-- ==== __Laws__
--
-- @
-- retry \`orElse\` m  ≡  m
-- m \`orElse\` retry  ≡  m
-- (m \`orElse\` n) \`orElse\` o  ≡  m \`orElse\` (n \`orElse\` o)
-- @
foreign import ccall "bhc_or_else"
    orElse :: STM a -> STM a -> STM a

-- | Retry if the condition is 'False'.
--
-- A convenient wrapper around 'retry' for condition checking.
--
-- ==== __Examples__
--
-- @
-- -- Withdraw only if sufficient balance
-- withdraw :: TVar Int -> Int -> STM ()
-- withdraw account amount = do
--     balance <- readTVar account
--     check (balance >= amount)  -- Retry if insufficient
--     writeTVar account (balance - amount)
-- @
--
-- This is equivalent to:
--
-- @
-- check b = if b then return () else retry
-- @
check :: Bool -> STM ()
check True  = return ()
check False = retry

-- | A transactional synchronization variable.
--
-- A 'TMVar' is either empty or contains a value. Taking from an empty
-- 'TMVar' blocks; putting to a full 'TMVar' blocks. This makes 'TMVar'
-- useful for:
--
-- * One-shot synchronization (futures/promises)
-- * Lock-like mutual exclusion
-- * Bounded channels
data TMVar a

-- | /O(1)/. Create a 'TMVar' containing a value.
--
-- >>> atomically $ newTMVar "hello" >>= takeTMVar
-- "hello"
foreign import ccall "bhc_new_tmvar"
    newTMVar :: a -> STM (TMVar a)

-- | /O(1)/. Create a 'TMVar' containing a value, in 'IO'.
--
-- >>> m <- newTMVarIO 42
-- >>> atomically $ takeTMVar m
-- 42
foreign import ccall "bhc_new_tmvar_io"
    newTMVarIO :: a -> IO (TMVar a)

-- | /O(1)/. Create an empty 'TMVar'.
--
-- >>> atomically $ do { m <- newEmptyTMVar; tryTakeTMVar m }
-- Nothing
foreign import ccall "bhc_new_empty_tmvar"
    newEmptyTMVar :: STM (TMVar a)

-- | /O(1)/. Create an empty 'TMVar' in 'IO'.
foreign import ccall "bhc_new_empty_tmvar_io"
    newEmptyTMVarIO :: IO (TMVar a)

-- | /O(1)/ to /O(∞)/. Take the value from a 'TMVar', leaving it empty.
--
-- If the 'TMVar' is empty, this retries until a value is available.
--
-- ==== __Examples__
--
-- @
-- -- Use as a future/promise
-- future <- newEmptyTMVarIO
-- forkIO $ do
--     result <- expensiveComputation
--     atomically $ putTMVar future result
-- -- Later...
-- result <- atomically $ takeTMVar future
-- @
foreign import ccall "bhc_take_tmvar"
    takeTMVar :: TMVar a -> STM a

-- | /O(1)/ to /O(∞)/. Put a value into a 'TMVar'.
--
-- If the 'TMVar' is full, this retries until it becomes empty.
--
-- ==== __Examples__
--
-- @
-- -- Use as a lock
-- lock <- newTMVarIO ()
-- atomically $ takeTMVar lock  -- Acquire
-- -- Critical section
-- atomically $ putTMVar lock ()  -- Release
-- @
foreign import ccall "bhc_put_tmvar"
    putTMVar :: TMVar a -> a -> STM ()

-- | /O(1)/ to /O(∞)/. Read the value without removing it.
--
-- Equivalent to taking and immediately putting back. If the 'TMVar'
-- is empty, this retries until a value is available.
readTMVar :: TMVar a -> STM a
readTMVar m = do
    x <- takeTMVar m
    putTMVar m x
    return x

-- | /O(1)/. Try to take the value, returning 'Nothing' if empty.
--
-- Non-blocking version of 'takeTMVar'.
tryTakeTMVar :: TMVar a -> STM (Maybe a)
tryTakeTMVar m = (Just <$> takeTMVar m) `orElse` return Nothing

-- | /O(1)/. Try to put a value, returning 'False' if full.
--
-- Non-blocking version of 'putTMVar'.
tryPutTMVar :: TMVar a -> a -> STM Bool
tryPutTMVar m x = (putTMVar m x >> return True) `orElse` return False

-- | /O(1)/. Check if a 'TMVar' is empty.
foreign import ccall "bhc_is_empty_tmvar"
    isEmptyTMVar :: TMVar a -> STM Bool

-- | /O(1)/ to /O(∞)/. Swap the value in a 'TMVar', returning the old value.
--
-- If the 'TMVar' is empty, this retries until a value is available.
swapTMVar :: TMVar a -> a -> STM a
swapTMVar m new = do
    old <- takeTMVar m
    putTMVar m new
    return old

-- | A transactional unbounded FIFO channel.
--
-- 'TChan' is an unbounded queue for communication between threads.
-- Multiple readers and writers are supported. Each reader sees all
-- messages written after they started reading.
--
-- ==== __Broadcast Pattern__
--
-- Use 'dupTChan' to create broadcast channels where each reader
-- receives all messages independently:
--
-- @
-- broadcast <- newTChanIO
-- reader1 <- atomically $ dupTChan broadcast
-- reader2 <- atomically $ dupTChan broadcast
-- atomically $ writeTChan broadcast "hello"
-- -- Both reader1 and reader2 will receive "hello"
-- @
data TChan a

-- | /O(1)/. Create a new empty 'TChan'.
foreign import ccall "bhc_new_tchan"
    newTChan :: STM (TChan a)

-- | /O(1)/. Create a new empty 'TChan' in 'IO'.
foreign import ccall "bhc_new_tchan_io"
    newTChanIO :: IO (TChan a)

-- | /O(1)/ to /O(∞)/. Read the next value from a 'TChan'.
--
-- If the channel is empty, this retries until a value is written.
-- Values are read in FIFO order.
--
-- ==== __Examples__
--
-- @
-- chan <- newTChanIO
-- forkIO $ atomically $ writeTChan chan "message"
-- msg <- atomically $ readTChan chan
-- @
foreign import ccall "bhc_read_tchan"
    readTChan :: TChan a -> STM a

-- | /O(1)/. Write a value to a 'TChan'.
--
-- Never blocks; the channel is unbounded.
foreign import ccall "bhc_write_tchan"
    writeTChan :: TChan a -> a -> STM ()

-- | /O(1)/. Duplicate a 'TChan'.
--
-- The duplicate channel starts empty but shares the write end.
-- Messages written after duplication appear in both channels.
--
-- ==== __Use Cases__
--
-- * Broadcast patterns (multiple readers, single writer)
-- * Pub/sub systems
-- * Event distribution
foreign import ccall "bhc_dup_tchan"
    dupTChan :: TChan a -> STM (TChan a)

-- | /O(1)/. Check if a 'TChan' is empty.
foreign import ccall "bhc_is_empty_tchan"
    isEmptyTChan :: TChan a -> STM Bool

-- | Throw an exception in STM, aborting the transaction.
--
-- The transaction is aborted and no changes are committed. The
-- exception propagates to the caller of 'atomically'.
--
-- ==== __Examples__
--
-- @
-- validateAndStore :: TVar Int -> Int -> STM ()
-- validateAndStore v x
--     | x < 0     = throwSTM (InvalidValue x)
--     | otherwise = writeTVar v x
-- @
--
-- ==== __Difference from 'error'__
--
-- Using 'throwSTM' is preferred over 'error' because it properly
-- participates in STM exception handling via 'catchSTM'.
foreign import ccall "bhc_throw_stm"
    throwSTM :: Exception e => e -> STM a

-- | Catch exceptions within a transaction.
--
-- If the action throws an exception of the specified type, the
-- transaction's effects are rolled back and the handler runs.
-- The handler's effects become the new transaction state.
--
-- ==== __Examples__
--
-- @
-- safeRead :: TVar Int -> STM Int
-- safeRead v = readTVar v \`catchSTM\` \\(e :: SomeException) -> return 0
-- @
--
-- ==== __Note__
--
-- 'catchSTM' cannot catch asynchronous exceptions or 'retry'.
-- For 'retry', use 'orElse' instead.
foreign import ccall "bhc_catch_stm"
    catchSTM :: Exception e => STM a -> (e -> STM a) -> STM a
