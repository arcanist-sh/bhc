-- A thread that never blocks is preempted at its IO steps, so the others run,
-- and an exception thrown to it reaches it (#75).
import Control.Concurrent
import Control.Exception
import Control.Monad
import Data.IORef

main :: IO ()
main = do
  r <- newIORef (0 :: Int)
  t <- forkIO $ forever $ modifyIORef' r (+ 1)
  threadDelay 50000
  n <- readIORef r
  putStrLn (if n > 0 then "busy thread ran" else "starved")
  done <- newEmptyMVar
  t2 <- forkIO $ (forever (modifyIORef' r (+ 1))) `finally` putMVar done ()
  threadDelay 20000
  killThread t2
  takeMVar done
  putStrLn "busy thread killed"
  killThread t
  putStrLn "end"
