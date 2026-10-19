-- Handle configuration is a no-op, but each call must still run as an IO
-- action in sequence (under the closure IO model it used to crash).
import System.IO

main :: IO ()
main = do
  putStrLn "start"
  hSetBuffering stdout NoBuffering
  hSetEncoding stdout utf8
  hSetNewlineMode stdout (NewlineMode LF LF)
  hSetBinaryMode stdout False
  hPutStrLn stdout "configured"
  putStrLn "done"
