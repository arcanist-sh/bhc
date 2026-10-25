-- `show` passed as a function value shows at the element type.
main :: IO ()
main = do
  print (map show [True, False])
  print (map show "ab")
  print (map show [1.5, 2.0 :: Double])
  print (map show [[1, 2], [3 :: Int]])
  print (map show [Just (1 :: Int), Nothing])
  print (map show [1, 2, 3 :: Int])
  mapM_ (putStrLn . show) [True, False]
