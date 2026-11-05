-- Chans (#38): a blocked reader, and a worker looping in `forever`.
import Control.Concurrent
import Control.Concurrent.Chan
import Control.Monad

main :: IO ()
main = do
  ch <- newChan
  res <- newEmptyMVar
  _ <- forkIO $ do
    xs <- replicateM 5 (readChan ch)
    putMVar res (sum xs :: Int)
  forM_ [1 .. 5] (writeChan ch)
  takeMVar res >>= print
  a <- newChan
  b <- newChan
  _ <- forkIO $ forever $ do
    n <- readChan a
    writeChan b (n * 2 :: Int)
  forM_ [1, 2, 3] $ \n -> do
    writeChan a n
    readChan b >>= print
