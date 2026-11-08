-- `mask $ \restore -> body` runs its body, and `restore` runs its action and
-- returns its result. Under the closure IO model the body only built an
-- action, which never ran, and an exception inside `restore` was lost (#91).
import Control.Concurrent
import Control.Exception

-- `forkFinally`, as `Control.Concurrent` defines it.
forkFinally' :: IO a -> (Either SomeException a -> IO ()) -> IO ThreadId
forkFinally' action andThen =
  mask $ \restore -> forkIO $ try (restore action) >>= andThen

main :: IO ()
main = do
  x <- mask $ \restore -> do
    putStrLn "a"
    y <- restore (putStrLn "b" >> return (41 :: Int))
    putStrLn "c"
    return (y + 1)
  print x
  mask_ (putStrLn "d")
  z <- uninterruptibleMask $ \restore -> restore (return (7 :: Int))
  print z
  r <- try (mask $ \restore -> restore (throwIO (userError "e")) >> putStrLn "not reached")
  print (either (\e -> show (e :: IOException)) (const "none") r)
  n <- mask $ \outer -> mask $ \inner -> inner (outer (return (3 :: Int)))
  print n
  done <- newEmptyMVar
  _ <- forkFinally' (return (10 :: Int)) $ \res -> do
    putStrLn (either (const "failed") (\v -> "finished with " ++ show v) res)
    putMVar done ()
  takeMVar done
  _ <- forkFinally' (throwIO (userError "boom") :: IO Int) $ \res -> do
    putStrLn (either (\e -> "failed: " ++ show e) (const "finished") res)
    putMVar done ()
  takeMVar done
  putStrLn "end"
