xs :: [Int]
xs = [1, 2, 3] ++ [4, 5]

ys :: [Int]
ys = reverse [1, 2, 3, 4]

joined :: [Int]
joined = [10] ++ [20] ++ [30]

main :: IO ()
main = do
  print xs
  print ys
  print joined
  print (length ([1, 2] ++ [3, 4, 5 :: Int]))
  putStrLn ("foo" ++ "bar")
