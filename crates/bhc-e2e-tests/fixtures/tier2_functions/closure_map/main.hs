applyN :: Int -> (Int -> Int) -> Int -> Int
applyN 0 _ x = x
applyN n f x = applyN (n - 1) f (f x)

main :: IO ()
main = do
  print (applyN 3 (\y -> y + 10) 0)
  print (applyN 4 (\y -> y * 2) 1)
