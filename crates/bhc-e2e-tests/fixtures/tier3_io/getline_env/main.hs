-- getLine and getEnv return Strings (#20, #21); lookupEnv returns Maybe.
import System.Environment (getEnv, lookupEnv)

main :: IO ()
main = do
  name <- getLine
  putStrLn ("hello, " ++ name)
  second <- getLine
  print (length second)
  putStrLn (reverse second)
  path <- getEnv "PATH"
  print (not (null path))
  unset <- lookupEnv "BHC_E2E_SURELY_UNSET_VARIABLE"
  print unset
  set <- lookupEnv "PATH"
  print (fmap null set)
