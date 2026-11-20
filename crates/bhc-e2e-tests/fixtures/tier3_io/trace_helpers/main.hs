-- Debug.Trace's helpers: traceShow, traceShowId, traceM and traceShowM write
-- to stderr, a value's `show` included, a user type's derived one too. In IO,
-- traceM's message comes out in sequence with the program's other effects.
import Debug.Trace
import qualified Debug.Trace as D
import System.IO

data Color = Red | Green deriving Show

next :: Int -> Int
next x = traceShow x (x + 1)

loop :: Int -> IO ()
loop 0 = pure ()
loop n = do
  traceM ("step " ++ show n)
  hPutStrLn stderr ("did " ++ show n)
  loop (n - 1)

main :: IO ()
main = do
  hSetBuffering stdout NoBuffering
  print (next 41)
  print (traceShowId (3 :: Int) + 1)
  hPutStrLn stderr "a"
  traceM "b"
  hPutStrLn stderr "c"
  loop 2
  traceShowM (Just Green)
  D.traceShowM [1, 2, 3 :: Int]
  print (Just (5 :: Int) >>= \v -> traceM ("maybe " ++ show v) >> pure (v * 2))
  print (D.traceShow ("pair", 'c') "done")
