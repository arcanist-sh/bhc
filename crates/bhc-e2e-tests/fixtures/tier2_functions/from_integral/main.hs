avg :: Int -> Int -> Double
avg a b = fromIntegral (a + b) / 2.0

main :: IO ()
main = do
  print (sqrt (fromIntegral (16 :: Int)))
  print (avg 3 5)
  print (fromIntegral (7 :: Int) + 0.5 :: Double)
  print (fromIntegral (length [1, 2, 3, 4]) * 1.5 :: Double)
