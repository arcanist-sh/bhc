-- A runtime list of Strings shows quoted on every backend (#41).
import System.Environment (getArgs)

main :: IO ()
main = do
  args <- getArgs
  print args
  print (words "a bb ccc")
  print (lines "one\ntwo")
  print (filter (/= "b") ["a", "b", "c"])
