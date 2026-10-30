-- [n ..] and [a, b ..] at Int are lazy, not capped at 10000 elements (#58).
main :: IO ()
main = do
  print ([1 :: Int ..] !! 20000, length (takeWhile (< 15000) [1 :: Int ..]))
  print (take 4 [10 :: Int, 7 ..], take 3 ['a' ..])
  print (zip [1 :: Int ..] "abc", lookup 3 (zip [1 :: Int ..] "xyz"))
  print (zipWith (+) [1 :: Int ..] [10, 20, 30])
  print (takeWhile (< 4) (map (* 2) [0 :: Int ..]))
  print (filter even (take 6 [1 :: Int ..]))
  print (head (dropWhile (< 12345) [1 :: Int ..]))
