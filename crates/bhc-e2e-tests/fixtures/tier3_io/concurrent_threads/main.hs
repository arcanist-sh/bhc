-- Scheduling (#38): delays wake in order, killThread stops a sleeping, blocked
-- or not yet started thread, and the program ends when main does. The two delays
-- are far apart because starting a thread can take tens of milliseconds on a
-- loaded machine, and the second delay only begins once its thread runs.
import Control.Concurrent
import Control.Monad

main :: IO ()
main = do
  out <- newChan
  _ <- forkIO (threadDelay 1000000 >> writeChan out "slow")
  _ <- forkIO (threadDelay 10000 >> writeChan out "fast")
  readChan out >>= putStrLn
  readChan out >>= putStrLn
  notStarted <- forkIO (threadDelay 100000 >> putStrLn "killed before it ran")
  killThread notStarted
  sleeping <- forkIO (threadDelay 100000 >> putStrLn "killed while sleeping")
  threadDelay 1000
  killThread sleeping
  blocked <- newEmptyMVar :: IO (MVar ())
  waiting <- forkIO (takeMVar blocked >> putStrLn "killed while blocked")
  threadDelay 1000
  killThread waiting
  putMVar blocked ()
  threadDelay 2000
  ms <- forM [1 .. 200 :: Int] $ \i -> do
    m <- newEmptyMVar
    _ <- forkIO (putMVar m (i * i))
    return m
  xs <- mapM takeMVar ms
  print (sum xs)
  _ <- forkIO (threadDelay 5000000 >> putStrLn "main ends first")
  putStrLn "done"
