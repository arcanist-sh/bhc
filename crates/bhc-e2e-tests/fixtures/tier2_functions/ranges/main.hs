sq :: Int -> Int
sq x = x * x

nums :: [Int]
nums = [1 .. 5]

main :: IO ()
main = do
  print (length [1 .. 6 :: Int])
  print (sum [1 .. 10])
  print nums
  print (foldl (+) 0 [1 .. 4])
  print (length (map sq [1 .. 4]))
