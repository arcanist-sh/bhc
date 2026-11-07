-- `throwTo` and `killThread` raise the exception in the other thread, whose
-- handlers run (#75). The other thread is blocked or sleeping when it is thrown
-- to, and its sleeps are far longer than the main thread's.
import Control.Concurrent
import Control.Exception
import Control.Monad

data Stop = Stop String deriving Show
instance Exception Stop

long :: Int
long = 5000000

main :: IO ()
main = do
  done <- newEmptyMVar
  -- `finally` runs in a killed thread.
  t1 <- forkIO $ (threadDelay long >> putStrLn "not reached")
    `finally` (putStrLn "cleanup" >> putMVar done ())
  threadDelay 10000
  killThread t1
  takeMVar done
  -- A `SomeException` handler sees `ThreadKilled`.
  gate <- newEmptyMVar :: IO (MVar ())
  t2 <- forkIO $ (takeMVar gate >> putStrLn "not reached") `catch` \e -> do
    putStrLn ("caught: " ++ show (e :: SomeException))
    putMVar done ()
  threadDelay 10000
  killThread t2
  takeMVar done
  -- `throwTo` with an `ErrorCall`, and with a user's exception type.
  t3 <- forkIO $ threadDelay long `catch` \e -> do
    putStrLn ("caught: " ++ show (e :: ErrorCall))
    putMVar done ()
  threadDelay 10000
  throwTo t3 (ErrorCall "boom")
  takeMVar done
  box <- newMVar (0 :: Int)
  t4 <- forkIO $ handle (\(Stop why) -> putStrLn ("stopped: " ++ why) >> putMVar done ()) $
    forever $ do
      modifyMVar_ box (return . (+ 1))
      threadDelay 1000
  threadDelay 20000
  throwTo t4 (Stop "enough")
  takeMVar done
  n <- readMVar box
  putStrLn (if n > 0 then "counted" else "no count")
  -- `bracket` releases in a killed thread.
  t5 <- forkIO $ bracket (putStrLn "acquire" >> return (5 :: Int))
    (\_ -> putStrLn "release" >> putMVar done ())
    (\x -> threadDelay long >> print x)
  threadDelay 10000
  killThread t5
  takeMVar done
  -- Threads blocked in `putMVar`, `readChan` and `readMVar` are taken out.
  full <- newMVar ()
  never <- newEmptyMVar :: IO (MVar ())
  ch <- newChan
  ws <- forM [1 :: Int, 2, 3] $ \i -> forkIO $
    (case i of
       1 -> putMVar full ()
       2 -> readChan ch >>= putStrLn
       _ -> readMVar never)
    `onException` (putStrLn ("unblocked " ++ show i) >> putMVar done ())
  threadDelay 10000
  forM_ ws $ \w -> killThread w >> takeMVar done
  writeChan ch "chan still works"
  _ <- forkIO (readChan ch >>= putStrLn >> putMVar done ())
  takeMVar done
  takeMVar full
  -- A thread that kills itself runs its handlers.
  _ <- forkIO $ ((myThreadId >>= killThread) >> putStrLn "not reached")
    `finally` (putStrLn "self-kill cleanup" >> putMVar done ())
  takeMVar done
  -- Killing a finished thread does nothing.
  t6 <- forkIO (putMVar done ())
  takeMVar done
  threadDelay 10000
  killThread t6
  -- A thread thrown to before it first ran ends at once.
  t7 <- forkIO (putStrLn "not reached")
  killThread t7
  threadDelay 10000
  putStrLn "end"
