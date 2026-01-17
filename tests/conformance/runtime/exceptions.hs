-- Test: exception-handling
-- Category: runtime
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 6.2

{-# HASKELL_EDITION 2026 #-}

module ExceptionTest where

import Control.Exception
import System.IO

-- Basic exception handling
test1 :: IO Int
test1 = catch
  (error "oops")
  (\(e :: SomeException) -> pure 42)
-- Result: 42

-- Specific exception types
data MyException = MyException String
  deriving (Show, Eq)

instance Exception MyException

test2 :: IO String
test2 = catch
  (throwIO (MyException "test"))
  (\(MyException msg) -> pure msg)
-- Result: "test"

-- Exception hierarchy
test3 :: IO String
test3 = catches
  (throwIO (MyException "test"))
  [ Handler (\(e :: MyException) -> pure "my exception")
  , Handler (\(e :: SomeException) -> pure "some exception")
  ]
-- Result: "my exception"

-- Finally (cleanup always runs)
test4 :: IO Int
test4 = do
  ref <- newIORef 0
  finally
    (writeIORef ref 1 >> error "oops")
    (writeIORef ref 42)
    `catch` \(e :: SomeException) -> pure ()
  readIORef ref
-- Result: 42 (cleanup ran)

-- Bracket for resource safety
test5 :: IO String
test5 = bracket
  (pure "resource")              -- acquire
  (\r -> putStrLn "released")   -- release (always runs)
  (\r -> pure r)                -- use
-- Result: "resource"

-- Bracket with exception
test6 :: IO Bool
test6 = do
  released <- newIORef False
  catch
    (bracket
      (pure ())
      (\_ -> writeIORef released True)
      (\_ -> error "oops"))
    (\(e :: SomeException) -> pure ())
  readIORef released
-- Result: True (release ran despite exception)

-- Async exceptions
test7 :: IO ()
test7 = do
  tid <- forkIO $ do
    threadDelay 1000000  -- 1 second
    pure ()
  threadDelay 100000     -- 0.1 seconds
  killThread tid         -- Throws ThreadKilled

-- Masking async exceptions
test8 :: IO Int
test8 = mask $ \restore -> do
  -- Critical section: async exceptions masked
  x <- pure 1
  y <- pure 2
  -- Allow exceptions in non-critical part
  z <- restore (pure 3)
  pure (x + y + z)
-- Result: 6

-- Try (returns Either)
test9 :: IO (Either SomeException Int)
test9 = try $ do
  x <- pure 1
  error "oops"
  pure x
-- Result: Left (SomeException ...)

-- Evaluate (force evaluation, catch exceptions)
test10 :: IO Int
test10 = do
  let x = error "lazy error"
  catch
    (evaluate x)
    (\(e :: SomeException) -> pure 42)
-- Result: 42

-- Exception in pure code
test11 :: Int
test11 =
  let x = error "pure error"
  in 42  -- x not evaluated
-- Result: 42

test12 :: Int
test12 =
  let x = error "pure error"
  in x + 1  -- Forces x
-- Result: throws exception

-- User-defined exception with context
data ValidationError = ValidationError
  { field :: String
  , message :: String
  }
  deriving (Show, Eq)

instance Exception ValidationError

test13 :: IO String
test13 = catch
  (throwIO $ ValidationError "email" "invalid format")
  (\(ValidationError f m) -> pure $ f ++ ": " ++ m)
-- Result: "email: invalid format"

-- Mock functions
newIORef :: a -> IO (IORef a)
newIORef = undefined

writeIORef :: IORef a -> a -> IO ()
writeIORef = undefined

readIORef :: IORef a -> IO a
readIORef = undefined

forkIO :: IO () -> IO ThreadId
forkIO = undefined

threadDelay :: Int -> IO ()
threadDelay = undefined

killThread :: ThreadId -> IO ()
killThread = undefined

evaluate :: a -> IO a
evaluate = undefined

data IORef a
data ThreadId
