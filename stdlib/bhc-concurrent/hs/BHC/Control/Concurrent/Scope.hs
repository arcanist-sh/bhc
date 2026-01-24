-- |
-- Module      : BHC.Control.Concurrent.Scope
-- Description : Structured concurrency primitives
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Structured concurrency ensures all spawned tasks complete before
-- their scope exits. This prevents task leaks and simplifies
-- resource management.
--
-- = Overview
--
-- Traditional concurrency with @forkIO@ allows spawned threads to outlive
-- their parent, leading to resource leaks and difficult-to-debug issues.
-- Structured concurrency enforces a simple rule: __all concurrent operations
-- happen within a scope that outlives them__.
--
-- = Quick Start
--
-- @
-- import BHC.Control.Concurrent.Scope
--
-- -- Parallel fetch with automatic cleanup
-- fetchBoth :: URL -> URL -> IO (Response, Response)
-- fetchBoth url1 url2 = withScope $ \\scope -> do
--     t1 <- spawn scope (fetch url1)
--     t2 <- spawn scope (fetch url2)
--     r1 <- await t1
--     r2 <- await t2
--     pure (r1, r2)
-- -- Both tasks guaranteed complete (or cancelled) before returning
-- @
--
-- = Task Lifecycle
--
-- @
-- spawn      await
--   │          │
--   ▼          ▼
-- ┌────┐    ┌─────────┐    ┌────────────┐    ┌───────────┐
-- │ New│ →  │ Running │ →  │ Completing │ →  │ Completed │
-- └────┘    └─────────┘    └────────────┘    └───────────┘
--              │                                   ▲
--              │ cancel                            │
--              ▼                                   │
--           ┌────────────┐                         │
--           │ Cancelling │ ────────────────────────┘
--           └────────────┘
-- @
--
-- = Cancellation
--
-- Cancellation is __cooperative__. Tasks check for cancellation at safe
-- points (between IO actions, at explicit checkpoints). When cancelled:
--
-- 1. The task is marked for cancellation
-- 2. All subtasks are also marked
-- 3. At the next safe point, 'Cancelled' is thrown
-- 4. Cleanup handlers registered with 'onCancel' run
--
-- = See Also
--
-- * 'BHC.Control.Concurrent.STM' for transactional memory
-- * H26-SPEC Section 10 for structured concurrency specification

module BHC.Control.Concurrent.Scope (
    -- * Scopes
    Scope,
    withScope,
    
    -- * Tasks
    Task,
    spawn,
    await,
    cancel,
    poll,
    
    -- * Deadlines
    withDeadline,
    withTimeout,
    
    -- * Cancellation
    checkCancelled,
    onCancel,
    Cancelled(..),
) where

import BHC.Prelude
import BHC.Data.Time (Duration)

-- | A scope for managing concurrent tasks.
--
-- A scope is a container that tracks all tasks spawned within it.
-- When the scope exits (via 'withScope'), it waits for all tasks
-- to complete, ensuring no task leaks.
data Scope = Scope
    { scopeId :: !Int
    , scopeTasks :: ![Task ()]
    }

-- | A handle to a spawned task.
--
-- A 'Task' represents a concurrent computation that may be running,
-- completed, or cancelled. Use 'await' to block for the result,
-- 'poll' to check non-blocking, or 'cancel' to request termination.
data Task a = Task
    { taskId :: !Int
    , taskResult :: !(IORef (Maybe (Either SomeException a)))
    }

-- | Exception thrown when a task is cancelled.
data Cancelled = Cancelled
    deriving (Show, Eq)

instance Exception Cancelled

-- | /O(1)/ setup, /O(n)/ cleanup. Run an action within a scope.
--
-- All tasks spawned in the scope must complete before this returns.
-- If the action throws an exception, all tasks are cancelled and
-- the exception is re-thrown after cleanup.
--
-- ==== __Examples__
--
-- >>> withScope $ \scope -> do
-- >>>     t1 <- spawn scope (pure 1)
-- >>>     t2 <- spawn scope (pure 2)
-- >>>     (+) <$> await t1 <*> await t2
-- 3
--
-- ==== __Exception Safety__
--
-- @
-- withScope $ \\scope -> do
--     spawn scope expensiveWork
--     throwIO SomeError        -- expensiveWork is cancelled
-- -- Scope waits for cancellation to complete before re-throwing
-- @
--
-- ==== __See Also__
--
-- * 'withDeadline' to add a time limit to a scope
-- * 'spawn' to create tasks within a scope
foreign import ccall "bhc_with_scope"
    withScope :: (Scope -> IO a) -> IO a

-- | /O(1)/. Spawn a task within a scope.
--
-- The task begins executing immediately on the work-stealing scheduler.
-- It inherits the scope's deadline (if any) and cancellation state.
--
-- ==== __Examples__
--
-- >>> withScope $ \scope -> do
-- >>>     t <- spawn scope (pure 42)
-- >>>     await t
-- 42
--
-- ==== __Inherited Properties__
--
-- Spawned tasks inherit:
--
-- * Deadline from parent scope (via 'withDeadline')
-- * Cancellation state (if scope is cancelled, task starts cancelled)
--
-- ==== __See Also__
--
-- * 'await' to wait for the task result
-- * 'cancel' to request task termination
foreign import ccall "bhc_spawn"
    spawn :: Scope -> IO a -> IO (Task a)

-- | /O(1)/ to /O(∞)/. Wait for a task to complete and return its result.
--
-- Blocks the current task until the awaited task completes. If the
-- awaited task threw an exception, it is re-thrown here. If the
-- current task is cancelled while waiting, 'Cancelled' is thrown.
--
-- ==== __Examples__
--
-- >>> withScope $ \scope -> do
-- >>>     t <- spawn scope (pure "hello")
-- >>>     await t
-- "hello"
--
-- ==== __Exception Propagation__
--
-- @
-- withScope $ \\scope -> do
--     t <- spawn scope (throwIO SomeError)
--     await t  -- throws SomeError
-- @
--
-- ==== __See Also__
--
-- * 'poll' for non-blocking check
-- * 'cancel' to stop a task before awaiting
foreign import ccall "bhc_await"
    await :: Task a -> IO a

-- | /O(1)/. Cancel a task and all its subtasks.
--
-- Marks the task for cancellation. The task will terminate at its
-- next safe point (between IO actions or at 'checkCancelled').
-- All subtasks spawned by this task are also cancelled.
--
-- This function returns immediately; use 'await' if you need to
-- wait for the task to actually finish.
--
-- ==== __Examples__
--
-- @
-- withScope $ \\scope -> do
--     t <- spawn scope longRunningTask
--     cancel t                     -- Request cancellation
--     await t \`catch\` \\Cancelled ->  -- Wait for completion
--         putStrLn "Task was cancelled"
-- @
--
-- ==== __Idempotence__
--
-- Calling 'cancel' multiple times on the same task is safe.
--
-- ==== __See Also__
--
-- * 'onCancel' to register cleanup handlers
-- * 'checkCancelled' to check cancellation in long loops
foreign import ccall "bhc_cancel"
    cancel :: Task a -> IO ()

-- | /O(1)/. Check if a task has completed without blocking.
--
-- Returns @Just result@ if the task has completed successfully,
-- or @Nothing@ if it is still running. If the task completed with
-- an exception, that exception is thrown.
--
-- ==== __Examples__
--
-- @
-- withScope $ \\scope -> do
--     t <- spawn scope (threadDelay 1000000 >> pure 42)
--     poll t >>= \\case
--         Nothing -> putStrLn "Still running..."
--         Just n  -> print n
-- @
--
-- ==== __Use Cases__
--
-- * Progress reporting in UI applications
-- * Implementing custom waiting strategies
-- * Non-blocking task completion checks
foreign import ccall "bhc_poll"
    poll :: Task a -> IO (Maybe a)

-- | /O(1)/ setup. Run an action with a deadline.
--
-- Returns @Just result@ if the action completes before the deadline,
-- or @Nothing@ if the deadline is exceeded. When the deadline passes,
-- the action is cancelled.
--
-- ==== __Examples__
--
-- >>> withDeadline (seconds 5) longOperation
-- Nothing  -- if it took more than 5 seconds
--
-- @
-- result <- withDeadline (seconds 10) $ withScope $ \\scope -> do
--     -- All spawned tasks inherit the 10-second deadline
--     t1 <- spawn scope fetchFromAPI
--     t2 <- spawn scope queryDatabase
--     (,) \<$\> await t1 \<*\> await t2
-- @
--
-- ==== __Deadline Inheritance__
--
-- Tasks spawned within a deadline scope inherit that deadline.
-- The effective deadline is the minimum of parent and child deadlines.
--
-- ==== __See Also__
--
-- * 'withTimeout' (alias for this function)
-- * 'withScope' for structured concurrency without deadline
foreign import ccall "bhc_with_deadline"
    withDeadline :: Duration -> IO a -> IO (Maybe a)

-- | /O(1)/ setup. Alias for 'withDeadline'.
--
-- Run an action with a timeout. Returns @Nothing@ if the timeout
-- expires before the action completes.
--
-- ==== __Examples__
--
-- >>> withTimeout (milliseconds 100) (threadDelay 1000000)
-- Nothing
--
-- >>> withTimeout (seconds 1) (pure 42)
-- Just 42
withTimeout :: Duration -> IO a -> IO (Maybe a)
withTimeout = withDeadline

-- | /O(1)/. Check if the current task has been cancelled.
--
-- If the current task has been marked for cancellation, this throws
-- 'Cancelled'. Otherwise, it returns immediately. Use this in long
-- loops to make them responsive to cancellation.
--
-- ==== __Examples__
--
-- @
-- processItems :: [Item] -> IO [Result]
-- processItems items = forM items $ \\item -> do
--     checkCancelled  -- Allow cancellation between items
--     processItem item
-- @
--
-- ==== __Safe Points__
--
-- The runtime automatically checks for cancellation at safe points:
--
-- * Between IO actions
-- * Before blocking operations
-- * At explicit 'checkCancelled' calls
--
-- For CPU-intensive loops without IO, add explicit checkpoints.
foreign import ccall "bhc_check_cancelled"
    checkCancelled :: IO ()

-- | /O(1)/ setup. Register a cleanup action to run on cancellation.
--
-- If the action is cancelled, the cleanup runs before 'Cancelled'
-- propagates. If the action completes normally, cleanup does not run.
--
-- ==== __Examples__
--
-- @
-- withTempFile $ \\path -> do
--     onCancel (processFile path) (removeFile path)
-- -- If cancelled, file is removed before exception propagates
-- @
--
-- ==== __Cleanup Guarantees__
--
-- * Cleanup runs exactly once if cancelled
-- * Cleanup does NOT run on normal completion
-- * Cleanup does NOT run on other exceptions (use 'bracket' for that)
--
-- ==== __See Also__
--
-- * 'Control.Exception.bracket' for general resource cleanup
-- * 'cancel' to trigger cancellation
foreign import ccall "bhc_on_cancel"
    onCancel :: IO a -> IO () -> IO a

-- Opaque types
data SomeException
data IORef a
