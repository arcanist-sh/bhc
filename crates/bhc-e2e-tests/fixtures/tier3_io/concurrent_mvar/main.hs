-- Threads and MVars (#38): values of each representation cross threads, and the
-- MVar operations keep their guarantees, including when an action throws.
import Control.Concurrent
import Control.Exception
import Control.Monad

worker :: MVar Int -> IO ()
worker mv = putMVar mv 42

main :: IO ()
main = do
  mv <- newEmptyMVar
  _ <- forkIO (worker mv)
  v <- takeMVar mv
  print (v + 1)
  s <- newEmptyMVar
  _ <- forkIO (putMVar s "hello")
  takeMVar s >>= putStrLn
  d <- newEmptyMVar
  _ <- forkIO (putMVar d (3.5 :: Double))
  x <- takeMVar d
  print (x * 2)
  l <- newMVar [1, 2, 3 :: Int]
  ys <- readMVar l
  print (sum ys)
  c <- newMVar (0 :: Int)
  done <- newEmptyMVar
  forM_ [1 .. 10 :: Int] $ \i -> forkIO $ do
    replicateM_ 100 (modifyMVar_ c (\n -> return (n + i)))
    putMVar done ()
  replicateM_ 10 (takeMVar done)
  readMVar c >>= print
  r <- modifyMVar c (\n -> return (n + 1, n * 2))
  print r
  w <- withMVar c (\n -> return (show n ++ "!"))
  putStrLn w
  old <- swapMVar c 7
  print old
  t <- tryTakeMVar c
  print t
  t2 <- tryTakeMVar c
  print t2
  p <- tryPutMVar c 9
  q <- tryPutMVar c 10
  print (p, q)
  catch
    (modifyMVar_ c (\_ -> ioError (userError "inner")))
    (\e -> const (putStrLn "caught") (e :: IOException))
  readMVar c >>= print
