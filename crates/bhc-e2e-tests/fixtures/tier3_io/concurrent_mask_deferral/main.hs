-- `mask` defers an exception thrown to a thread until it unmasks or blocks,
-- `uninterruptibleMask` until it unmasks, and `throwTo` waits meanwhile (#75).
import Control.Concurrent
import Control.Exception
import Control.Monad
import Data.IORef

main :: IO ()
main = do
  done <- newEmptyMVar
  never <- newEmptyMVar :: IO (MVar ())
  -- uninterruptibleMask_ defers ThreadKilled across a blocking threadDelay;
  -- killThread waits until it is delivered.
  t1 <- forkIO $ (uninterruptibleMask_ (threadDelay 50000 >> putStrLn "masked sleep done")
                  >> putStrLn "not reached 1") `finally` putMVar done ()
  threadDelay 10000
  killThread t1
  putStrLn "after kill 1"
  takeMVar done
  -- mask_ defers it across work that does not block.
  r <- newIORef (0 :: Int)
  t2 <- forkIO $ (mask_ (forM_ [1 .. 20000000 :: Int] (\i -> modifyIORef' r (+ i))
                         >> putStrLn "masked work done")
                  >> takeMVar never) `finally` putMVar done ()
  threadDelay 1000
  killThread t2
  putStrLn "after kill 2"
  takeMVar done
  -- Under mask_, blocking is interruptible: it is delivered at once.
  t3 <- forkIO $ mask_ (takeMVar never) `catch` \e -> do
    putStrLn ("interrupted while masked: " ++ show (e :: AsyncException))
    putMVar done ()
  threadDelay 10000
  killThread t3
  takeMVar done
  -- An exception deferred by mask is raised when the thread unmasks.
  t4 <- forkIO $ (mask $ \restore -> do
                    forM_ [1 .. 20000000 :: Int] (\i -> modifyIORef' r (+ i))
                    putStrLn "masked part ran"
                    restore (putStrLn "not reached 4"))
                 `catch` \e -> do
                   putStrLn ("raised on unmask: " ++ show (e :: AsyncException))
                   putMVar done ()
  threadDelay 1000
  killThread t4
  putStrLn "after kill 4"
  takeMVar done
  putStrLn "end"
