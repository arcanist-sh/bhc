-- `mask` gives its body a polymorphic `restore` (`forall a. IO a -> IO a`),
-- used here at `IO ()` and at `IO a`; and `showFFloat` takes any RealFloat.
-- Both were typed too narrowly, which the check that a signature's type
-- variables stay general turned into errors in pandoc's Text.Pandoc.Process
-- and Text.Pandoc.ImageSize (#97).
import qualified Control.Exception as E
import Control.Concurrent
import Numeric (showFFloat)

withForkWait :: IO () -> (IO () -> IO a) -> IO a
withForkWait async body = do
  waitVar <- newEmptyMVar :: IO (MVar (Either E.SomeException ()))
  E.mask $ \restore -> do
    tid <- forkIO $ E.try (restore async) >>= putMVar waitVar
    let wait = takeMVar waitVar >>= either E.throwIO return
    restore (body wait) `E.onException` killThread tid

showFl :: RealFloat a => a -> String
showFl a = showFFloat (Just 2) a ""

main :: IO ()
main = do
  r <- withForkWait (putStrLn "async") (\wait -> wait >> return (42 :: Int))
  print r
  putStrLn (showFl (3.14159 :: Double))
