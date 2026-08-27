module Main where

-- `concat` returns its argument's ELEMENT type, so `concat ["x","y"]` is a
-- String. Show inference looked at the whole argument instead, decided
-- list-of-lists, and printed the result as character codes.
main :: IO ()
main = do
  print (concat ["x", "y", "z"])
  print ("xy" ++ "z")
  print (concat [[1], [2, 3 :: Int]])
  putStrLn (concat ["a", "b"])
