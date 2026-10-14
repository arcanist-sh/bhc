module Main where

-- `print` of a case-alternative / do-bound binder: Core carries the binder
-- untyped, so the show has to come from the type typeck solved for it.

get1 :: IO (Either String [Int])
get1 = return (Right [1, 2])

wrap :: IO a -> IO (Either String a)
wrap m = fmap Right m

main :: IO ()
main = do
  r <- get1
  case r of { Left _ -> putStrLn "L"; Right ns -> print ns }
  r2 <- wrap (return [3, 4 :: Int])
  case r2 of { Left _ -> putStrLn "L"; Right ns -> print ns }
  case Just (True, 'x') of { Nothing -> putStrLn "N"; Just p -> print p }
