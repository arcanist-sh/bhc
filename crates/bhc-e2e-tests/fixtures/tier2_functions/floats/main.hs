addD :: Double -> Double -> Double
addD x y = x + y

mulD :: Double -> Double -> Double
mulD x y = x * y

main :: IO ()
main = do
  print (3.5 :: Double)
  print (addD 3.5 2.25)
  print (mulD 1.5 4.0)
  print (7.0 / 2.0 :: Double)
  print (10.0 - 3.5 :: Double)
  print (negate 2.5 :: Double)
  print (3.5 < 4.0)
