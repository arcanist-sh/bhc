-- A builtin operator used as a function VALUE and applied to fewer arguments
-- than it takes: `fmap`/`<$>` must build a partial application, not run the
-- operator with a missing argument.
module Main where

main :: IO ()
main = do
  let g = fmap (++) (Just "p")
  case g of
    Just h -> putStrLn (h "q")
    Nothing -> return ()
  h2 <- (++) <$> return "a"
  putStrLn (h2 "b")
  s3 <- (++) <$> return "c" <*> return "d"
  putStrLn s3
  s4 <- (\a b -> a ++ b) <$> return "e" <*> return "f"
  putStrLn s4
