-- withFile opens a file, runs the action on its handle and closes the handle,
-- whether the action returns or throws. It had a type and no implementation:
-- a program that used it stopped at the call.
import Control.Exception
import System.IO
import qualified System.IO as IO

main :: IO ()
main = do
  withFile "bhc_e2e_with_file.txt" WriteMode (\h -> hPutStrLn h "one" >> hPutStrLn h "two")
  n <- withFile "bhc_e2e_with_file.txt" ReadMode $ \h -> do
    a <- hGetLine h
    b <- hGetLine h
    return (length a + length b)
  print n
  IO.withFile "bhc_e2e_with_file.txt" AppendMode (`hPutStrLn` "three")
  s <- readFile "bhc_e2e_with_file.txt"
  putStr s
  let writer = withFile "bhc_e2e_with_file2.txt" WriteMode
  writer (\h -> hPutStr h "partial")
  readFile "bhc_e2e_with_file2.txt" >>= putStrLn
  r <- try (withFile "bhc_e2e_with_file.txt" ReadMode (\h -> hGetLine h >> throwIO (userError "inside")))
  case r of
    Left e -> putStrLn ("caught: " ++ show (e :: IOException))
    Right () -> putStrLn "no exception"
  r2 <- try (withFile "/nonexistent/dir/x" ReadMode hGetLine)
  case r2 of
    Left e -> putStrLn ("open failed: " ++ show (isDoesNotExist e))
    Right l -> putStrLn l
 where
  isDoesNotExist :: IOException -> Bool
  isDoesNotExist _ = True
