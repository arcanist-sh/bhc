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
data Scope = Scope
    { scopeId :: !Int
    , scopeTasks :: ![Task ()]
    }

-- | A handle to a spawned task.
data Task a = Task
    { taskId :: !Int
    , taskResult :: !(IORef (Maybe (Either SomeException a)))
    }

-- | Exception thrown when a task is cancelled.
data Cancelled = Cancelled
    deriving (Show, Eq)

instance Exception Cancelled

-- | Run an action within a scope.
-- All tasks spawned in the scope must complete before this returns.
--
-- > withScope $ \scope -> do
-- >     t1 <- spawn scope computation1
-- >     t2 <- spawn scope computation2
-- >     r1 <- await t1
-- >     r2 <- await t2
-- >     pure (r1, r2)
foreign import ccall "bhc_with_scope"
    withScope :: (Scope -> IO a) -> IO a

-- | Spawn a task within a scope.
-- The task inherits the scope's deadline.
foreign import ccall "bhc_spawn"
    spawn :: Scope -> IO a -> IO (Task a)

-- | Wait for a task to complete and return its result.
-- Throws if the task threw an exception.
foreign import ccall "bhc_await"
    await :: Task a -> IO a

-- | Cancel a task.
-- Subtasks are also cancelled.
foreign import ccall "bhc_cancel"
    cancel :: Task a -> IO ()

-- | Check if a task has completed without blocking.
foreign import ccall "bhc_poll"
    poll :: Task a -> IO (Maybe a)

-- | Run with a deadline.
-- Returns Nothing if the deadline is exceeded.
foreign import ccall "bhc_with_deadline"
    withDeadline :: Duration -> IO a -> IO (Maybe a)

-- | Run with a timeout.
-- Returns Nothing if the timeout expires.
withTimeout :: Duration -> IO a -> IO (Maybe a)
withTimeout = withDeadline

-- | Check if the current task has been cancelled.
-- Throws 'Cancelled' if so.
foreign import ccall "bhc_check_cancelled"
    checkCancelled :: IO ()

-- | Register a cleanup action to run on cancellation.
foreign import ccall "bhc_on_cancel"
    onCancel :: IO a -> IO () -> IO a

-- Opaque types
data SomeException
data IORef a
