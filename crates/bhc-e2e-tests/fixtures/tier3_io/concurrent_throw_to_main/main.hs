-- The main thread can be thrown to by another thread, and an uncaught
-- `ThreadKilled` ends the program with status 1 (#75).
import Control.Concurrent
import Control.Exception

main :: IO ()
main = do
  me <- myThreadId
  gate <- newEmptyMVar :: IO (MVar ())
  _ <- forkIO (throwTo me (ErrorCall "from child"))
  r <- try (takeMVar gate)
  case r of
    Left e -> putStrLn ("main caught: " ++ show (e :: ErrorCall))
    Right () -> return ()
  _ <- forkIO (killThread me)
  takeMVar gate `catch` \e -> putStrLn ("async: " ++ show (e :: AsyncException))
  (myThreadId >>= killThread) `finally` putStrLn "main cleanup"
  putStrLn "not reached"
