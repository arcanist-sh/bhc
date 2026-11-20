-- A file operation that fails raises an IOException. openFile, writeFile and
-- appendFile used to carry on as if they had succeeded (openFile returned a
-- handle that was no handle), so `try` reported success and nothing told the
-- program its file was not written.
import System.IO
import Control.Exception
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Lazy as BL

report :: String -> Either IOException a -> IO ()
report what (Left e) = putStrLn (what ++ ": " ++ show e)
report what (Right _) = putStrLn (what ++ ": no exception")

main :: IO ()
main = do
  r1 <- try (readFile "/nonexistent-bhc/a.txt") :: IO (Either IOException String)
  report "readFile" r1
  r2 <- try (openFile "/nonexistent-bhc/b.txt" ReadMode) :: IO (Either IOException Handle)
  report "openFile" r2
  r3 <- try (writeFile "/nonexistent-bhc/c.txt" "x") :: IO (Either IOException ())
  report "writeFile" r3
  r4 <- try (appendFile "/nonexistent-bhc/d.txt" "x") :: IO (Either IOException ())
  report "appendFile" r4
  r5 <- try (openFile "/nonexistent-bhc/e.txt" ReadMode >>= hGetLine) :: IO (Either IOException String)
  report "openFile then hGetLine" r5
  (openFile "/nonexistent-bhc/f.txt" WriteMode >> putStrLn "not reached")
    `catch` \e -> putStrLn ("caught: " ++ show (e :: IOException))
  -- The operations still work when they can.
  writeFile "bhc_e2e_io_errors.txt" "one\n"
  appendFile "bhc_e2e_io_errors.txt" "two\n"
  s <- readFile "bhc_e2e_io_errors.txt"
  putStr s
  h <- openFile "bhc_e2e_io_errors.txt" ReadMode
  l <- hGetLine h
  hClose h
  putStrLn l
  -- The Text and ByteString file functions, which live in another library.
  t1 <- try (TIO.readFile "/nonexistent-bhc/g.txt") :: IO (Either IOException T.Text)
  report "TIO.readFile" t1
  t2 <- try (TIO.writeFile "/nonexistent-bhc/h.txt" (T.pack "x")) :: IO (Either IOException ())
  report "TIO.writeFile" t2
  t3 <- try (TIO.appendFile "/nonexistent-bhc/i.txt" (T.pack "x")) :: IO (Either IOException ())
  report "TIO.appendFile" t3
  b1 <- try (BS.readFile "/nonexistent-bhc/j.bin") :: IO (Either IOException BS.ByteString)
  report "BS.readFile" b1
  b2 <- try (BS.writeFile "/nonexistent-bhc/k.bin" (BC.pack "x")) :: IO (Either IOException ())
  report "BS.writeFile" b2
  b3 <- try (BL.readFile "/nonexistent-bhc/l.bin") :: IO (Either IOException BL.ByteString)
  report "BL.readFile" b3
  TIO.writeFile "bhc_e2e_io_errors.txt" (T.pack "text\n")
  t <- TIO.readFile "bhc_e2e_io_errors.txt"
  TIO.putStr t
