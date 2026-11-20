-- The WASM backend has no System.Directory. It used to compile the call to
-- the constant 0, so this printed "0"; now the build fails and names the
-- function (#99).
import System.Directory (doesFileExist)

main :: IO ()
main = do
  b <- doesFileExist "nope.txt"
  print b
