-- Pure code is preempted on function entry and at allocations, so a thread
-- in a long pure computation does not keep the others from running, and
-- killThread reaches it at its next IO step (#75).
import Control.Concurrent
import Control.Exception
import Data.List (foldl')

-- A tail-recursive loop over unboxed Ints: it neither allocates nor reaches
-- an IO step.
go :: Int -> Int -> Int
go 0 acc = acc
go n acc = go (n - 1) (acc + n)

-- Non-tail recursion, which allocates nothing either.
fib :: Int -> Int
fib k = if k < 2 then k else fib (k - 1) + fib (k - 2)

main :: IO ()
main = do
  -- Read inside each thread, so the work cannot happen before the fork.
  nref <- newMVar (20000000 :: Int)
  done <- newEmptyMVar
  _ <- forkIO $ do
    n <- readMVar nref
    let s = go (n * 50) 0
    s `seq` putStrLn ("loop done: " ++ show s)
    putMVar done ()
  threadDelay 1000
  putStrLn "main ran during the loop"
  takeMVar done
  _ <- forkIO $ do
    n <- readMVar nref
    let s = fib (n `div` 500000)
    s `seq` putStrLn ("fib done: " ++ show s)
    putMVar done ()
  threadDelay 1000
  putStrLn "main ran during fib"
  takeMVar done
  _ <- forkIO $ do
    n <- readMVar nref
    let s = foldl' (+) 0 [1 .. n * 5]
    s `seq` putStrLn ("fold done: " ++ show s)
    putMVar done ()
  threadDelay 1000
  putStrLn "main ran during the fold"
  takeMVar done
  -- killThread is raised at the thread's next IO step after the loop.
  t <- forkIO $ (do
                   n <- readMVar nref
                   let s = go (n * 50) 0
                   s `seq` return ()
                   putStrLn "not reached")
                `catch` \e -> do
                  putStrLn ("killed: " ++ show (e :: AsyncException))
                  putMVar done ()
  threadDelay 1000
  killThread t
  takeMVar done
  putStrLn "end"
