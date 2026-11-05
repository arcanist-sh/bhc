-- `try` catches at the type it is used at, and an exception shows as GHC's does
-- (#70, #72). `evaluate` forces inside the `try` (#71).
import Control.Exception

main :: IO ()
main = do
  a <- try (throwIO (userError "x")) :: IO (Either IOException ())
  putStrLn (either (const "left") (const "right") a)
  b <- try (return 5) :: IO (Either IOException Int)
  print (either (const 0) id b)
  e <- try (ioError (userError "y")) :: IO (Either IOException ())
  putStrLn (either show (const "right") e)
  -- GHC's text for this also has a call stack, which bhc does not record.
  c <- try (evaluate (error "bad" :: Int)) :: IO (Either ErrorCall Int)
  putStrLn (either (const "error caught") show c)
  d <- try (evaluate (2 + 3 :: Int)) :: IO (Either ErrorCall Int)
  putStrLn (either show show d)
  f <- try (ioError (userError "z")) :: IO (Either SomeException ())
  putStrLn (either show (const "right") f)
