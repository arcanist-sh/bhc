-- Builtins passed to higher-order functions rather than applied directly.
-- Nearly every builtin is implemented only for the expression position, so
-- reaching one as a value used to abort with `stub: <name> not implemented`.
module Main where

main :: IO ()
main = do
  print (map length ["ab", "c"])
  putStrLn (head (map reverse ["ab", "cd"]))
  putStrLn (concat (map reverse ["ab", "cd"]))
  print (length (filter null ["", "a", ""]))
  print (sum (map length ["ab", "c"]))
  putStrLn (unwords (map reverse ["ab", "cd"]))
  print (map fromEnum "ab")
  print (foldr (+) 0 [1, 2, 3 :: Int])
  mapM_ putStrLn (map unwords [["a", "b"]])
