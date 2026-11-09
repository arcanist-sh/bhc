-- `mask $ \restore -> body` and `uninterruptibleMask` run their body once,
-- with `restore` returning its action's result, in both IO models. Under the
-- eager model the body was called with its arguments swapped, so it ran
-- twice and `restore` crashed (#91).
import Control.Exception

main :: IO ()
main = do
  x <- mask $ \restore -> do
    putStrLn "a"
    y <- restore (putStrLn "b" >> return (41 :: Int))
    putStrLn "c"
    return (y + 1)
  print x
  mask_ (putStrLn "d")
  z <- uninterruptibleMask $ \restore -> restore (return (7 :: Int))
  print z
  r <- try (mask $ \restore -> restore (throwIO (userError "e")) >> putStrLn "not reached")
  print (either (\e -> show (e :: IOException)) (const "none") r)
  n <- mask $ \outer -> mask $ \inner -> inner (outer (return (3 :: Int)))
  print n
  putStrLn "end"
