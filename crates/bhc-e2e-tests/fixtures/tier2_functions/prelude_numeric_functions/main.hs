-- Prelude numeric functions that were stubs, unbound, or wrong at Integer:
-- (**), (^^), realToFrac, logBase, the inverse and hyperbolic functions,
-- isNaN/isInfinite, and floor/ceiling/round/truncate at Integer.
module Main where

hyp :: Double -> Double -> Double
hyp a b = sqrt (a ** 2 + b ** 2)

main :: IO ()
main = do
  print (2 ** 3 :: Double, 2 ** 0.5 :: Double)
  print (2 ^^ (-1 :: Int) :: Double, 10 ^^ (3 :: Int) :: Double)
  print (hyp 3 4)
  print (realToFrac (1.5 :: Double) :: Double)
  print (realToFrac (3 :: Int) :: Double)
  print (realToFrac (1.5 :: Float) + (1 :: Double))
  print (logBase 2 8 :: Double, logBase 10 1000 :: Double)
  print (sinh 1 :: Double, cosh 1 :: Double, tanh 1 :: Double)
  print (asinh 1 :: Double, acosh 2 :: Double, atanh 0.5 :: Double)
  print (asin 1 :: Double, acos 1 :: Double, atan 1 :: Double, atan2 1 1 :: Double)
  print (isNaN (0 / 0 :: Double), isNaN (1 :: Double), isInfinite (1 / 0 :: Double), isInfinite (1 :: Double))
  print (floor (2.7 :: Double) :: Integer, ceiling (2.1 :: Double) :: Integer)
  print (round (2.5 :: Double) :: Integer, round (3.5 :: Double) :: Integer, truncate (-2.7 :: Double) :: Integer)
  print (floor (2.7 :: Double) :: Int, ceiling (2.1 :: Double) :: Int, round (2.5 :: Double) :: Int)
  print (floor (1.0e20 :: Double) :: Integer)
  print (map floor [1.5, -2.5 :: Double] :: [Integer])
  print (map (** 2) [1, 2, 3 :: Double], map realToFrac [1, 2 :: Int] :: [Double])
  print (zipWith (**) [2, 3 :: Double] [2, 2], map sinh [0 :: Double])
