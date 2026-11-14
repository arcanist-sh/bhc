-- A thread's id is equal to itself. Each `myThreadId` used to make a new
-- object and `==` compared the objects, so `me == me` was False and a
-- ThreadId was never found in a list.
import Control.Concurrent
import Control.Monad
main :: IO ()
main = do
  me <- myThreadId
  me2 <- myThreadId
  print (me == me2, me /= me2)
  done <- newEmptyMVar
  inner <- newEmptyMVar
  t <- forkIO $ do
    self <- myThreadId
    putMVar inner self
    putMVar done ()
  takeMVar done
  self <- takeMVar inner
  print (t == self, t == me, self /= me)
  ts <- forM [1 .. 3 :: Int] $ \_ -> forkIO (return ())
  print (me `elem` ts, t `elem` (t : ts), length (filter (== me) (me : ts)))
