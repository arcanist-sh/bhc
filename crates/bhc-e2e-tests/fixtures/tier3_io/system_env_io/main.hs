-- Zero-argument system IO actions run as actions, in sequence.
import System.Directory (getCurrentDirectory)
import System.Environment (getArgs, getProgName)

main :: IO ()
main = do
  args <- getArgs
  print (length args)
  name <- getProgName
  putStrLn (if null name then "no name" else "has name")
  dir <- getCurrentDirectory
  putStrLn (if null dir then "no dir" else "has dir")
