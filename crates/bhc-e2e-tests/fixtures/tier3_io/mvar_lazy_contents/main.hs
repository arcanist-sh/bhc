-- MVars and Chans are lazy in their contents: storing a value does not
-- evaluate it, and a bottom nobody reads never fails. The reader evaluates a
-- stored computation, in the reading thread.
import Control.Concurrent

expensive :: Int -> Int
expensive n = sum [1 .. n]

main :: IO ()
main = do
  _ <- newMVar (error "never read" :: Int)
  putStrLn "newMVar of a bottom"
  m <- newMVar (expensive 100)
  takeMVar m >>= print
  s <- newMVar (show (expensive 10))
  readMVar s >>= putStrLn
  readMVar s >>= putStrLn
  xs <- newMVar (map (* 2) [1, 2, 3 :: Int])
  ys <- takeMVar xs
  print (sum ys)
  pair <- newMVar (length "hello", 'c')
  (n, c) <- takeMVar pair
  print n
  print c
  e <- newEmptyMVar
  putMVar e (error "never read either" :: Int)
  full <- tryPutMVar e 3
  print full
  putMVar m (expensive 4)
  old <- swapMVar m (expensive 5)
  print old
  mt <- tryTakeMVar m
  case mt of
    Just v -> print v
    Nothing -> putStrLn "empty"
  done <- newEmptyMVar
  _ <- forkIO (putMVar done (expensive 1000))
  takeMVar done >>= print
  ch <- newChan
  writeChan ch (expensive 6)
  writeChan ch (error "never read from the Chan" :: Int)
  readChan ch >>= print
  putStrLn "done"
