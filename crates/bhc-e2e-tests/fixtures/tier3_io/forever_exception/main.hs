-- `forever` runs its body under the closure IO model, and a throw ends it: the
-- actions after the throw do not run.
import Control.Exception
import Control.Monad
import Data.IORef

main :: IO ()
main = do
  r <- newIORef (0 :: Int)
  catch
    (forever (do
      n <- readIORef r
      when (n > 2) (ioError (userError "stop"))
      print n
      writeIORef r (n + 1)))
    (\e -> putStrLn "stopped" >> const (return ()) (e :: IOException))
  readIORef r >>= print
