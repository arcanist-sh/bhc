-- Test: structured-concurrency
-- Category: runtime
-- Profile: server
-- Expected: success
-- Spec: H26-SPEC Section 10.1

{-# PROFILE Server #-}

module ConcurrencyTest where

import H26.Concurrency

-- Tasks must complete or cancel before scope exits
test1 :: IO Int
test1 = withScope $ \scope -> do
  t1 <- spawn scope (pure 1)
  t2 <- spawn scope (pure 2)
  x <- await t1
  y <- await t2
  pure (x + y)  -- Result: 3

-- Cancellation propagates to children
test2 :: IO ()
test2 = withScope $ \scope -> do
  t1 <- spawn scope $ withScope $ \inner -> do
    t2 <- spawn inner longRunningTask
    await t2
  -- Cancelling t1 should also cancel t2
  cancel t1

-- Deadlines propagate to child tasks
test3 :: IO (Maybe Int)
test3 = withDeadline (seconds 5) $ withScope $ \scope -> do
  t1 <- spawn scope expensiveComputation  -- Inherits 5s deadline
  await t1

-- Mock functions (to be implemented)
longRunningTask :: IO ()
longRunningTask = pure ()

expensiveComputation :: IO Int
expensiveComputation = pure 42
