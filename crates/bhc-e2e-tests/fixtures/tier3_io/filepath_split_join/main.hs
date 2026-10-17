-- splitPath / splitDirectories / joinPath with System.FilePath.Posix semantics
-- (pandoc's makeCanonical round-trips data-file paths through them).
module Main where

import System.FilePath (joinPath, splitDirectories, splitPath, (</>))
import qualified System.FilePath.Posix as Posix

main :: IO ()
main = do
  print (splitDirectories "/usr/local/bin")
  print (splitDirectories "a//b/")
  print (splitPath "/usr/local/bin")
  print (splitPath "a//b/")
  putStrLn (joinPath ["/", "usr", "bin"])
  putStrLn (joinPath ["a/", "b", "c.txt"])
  putStrLn (joinPath ["a", "/abs", "c"])
  putStrLn (Posix.joinPath (Posix.splitDirectories "data/translations/en.yaml"))
  print (length (splitDirectories ("x" </> "y" </> "z")))
  print (map length (splitPath "ab/cde/f"))
