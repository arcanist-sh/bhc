dist :: Double -> Double -> Double
dist x y = sqrt (x * x + y * y)

main :: IO ()
main = do
  print (dist 3.0 4.0)
  print (sqrt (2.0 :: Double))
  print (truncate (3.7 :: Double) :: Int)
  print (floor (3.7 :: Double) :: Int)
  print (ceiling (3.2 :: Double) :: Int)
  print (round (3.5 :: Double) :: Int)
  print (abs (negate 2.5 :: Double))
