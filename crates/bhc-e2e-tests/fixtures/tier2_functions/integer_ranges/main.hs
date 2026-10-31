-- Ranges and list consumers at Integer (boxed bignums), #51.
main :: IO ()
main = do
  let xs = [1, 2, 3, 4] :: [Integer]
      big = 2 ^ 64 :: Integer
  print [1 .. 3 :: Integer]
  print (length [1 .. 5 :: Integer], sum [1 .. 10 :: Integer])
  print [1, 3 .. 9 :: Integer]
  print [10, 8 .. 1 :: Integer]
  print (take 3 [10 :: Integer ..])
  print (take 3 [big, big + 5 ..])
  print [big - 1 .. big + 1]
  print (even (2 :: Integer), odd (3 :: Integer))
  print (filter even xs, filter odd [1 .. 9 :: Integer])
  print (minimum [3, 1, 2 :: Integer], maximum [1, 3, 2 :: Integer])
  print [x * x | x <- [1 .. 4 :: Integer]]
  print (map (* 2) xs, Just (10 :: Integer))
  print (zip xs "abc")
  print [5 .. 1 :: Integer]
