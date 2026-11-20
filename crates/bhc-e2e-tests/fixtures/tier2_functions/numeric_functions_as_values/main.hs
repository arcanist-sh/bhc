-- Numeric conversions and operators passed as function values, and unary
-- negation of a Double variable (#100).
module Main where

import Data.List (foldl')

mean :: [Int] -> Double
mean xs = sum (map fromIntegral xs) / fromIntegral (length xs)

applyTo :: (Int -> Double) -> Int -> Double
applyTo f n = f n + 0.5

neg :: Double -> Double
neg x = negate x

main :: IO ()
main = do
  print (sum (map (\x -> x * 2.5) (map fromIntegral [1..10 :: Int])) :: Double)
  print (sum (map fromIntegral [1..10 :: Int]) :: Double)
  print (map fromIntegral [1..3 :: Int] :: [Double])
  let xs = map fromIntegral [1..4 :: Int] :: [Double]
  print (sum xs / fromIntegral (length xs))
  print (mean [1, 2, 3, 4])
  print (applyTo fromIntegral 7)
  print (foldl' (+) 0 (map fromIntegral [1..5 :: Int]) :: Double)
  print (map (fromIntegral . (* 2)) [1..3 :: Int] :: [Double])
  print (zipWith (/) (map fromIntegral [1, 2, 3 :: Int]) [2, 4, 8 :: Double])
  print (map fromIntegral [1..3 :: Int] :: [Integer])
  print (map toInteger [4, 5 :: Int])
  print (sum (map fromIntegral [1..3 :: Int]) :: Int)
  print (map fromIntegral [10, 20 :: Integer] :: [Double])
  print (fmap fromIntegral (Just (3 :: Int)) :: Maybe Double)
  print (zipWith (/) (map fromIntegral [1, 2, 3 :: Int]) [2, 4, 8 :: Double])
  print (zipWith (+) (map fromIntegral [1, 2, 3 :: Int]) [2, 4, 8 :: Double])
  print (zipWith (/) (map (* 2) [1, 2, 3 :: Double]) [2, 4, 8 :: Double])
  print (foldr (/) 1 [8, 4, 2 :: Double])
  print (foldr (-) 1 [8, 4, 2 :: Double])
  print (foldl (/) 64 [8, 4, 2 :: Double])
  print (map negate [1.5, 2.5 :: Double])
  print (map abs [-1.5, 2.5 :: Double])
  print (map signum [-1.5, 0, 2.5 :: Double])
  print (neg 1.5)
  print (map (\x -> negate x) [1.5, 2.5 :: Double])
  print (map (\x -> abs x) [-1.5, 2.5 :: Double])
  print (map (\x -> signum x) [-1.5, 2.5 :: Double])
  print (map (\x -> -x) [1.5, 2.5 :: Double])
  print (map negate [3, -4 :: Int], map abs [3, -4 :: Int], map signum [3, -4, 0 :: Int])
  print (negate (2 :: Integer) ^ (70 :: Int), abs (-5 :: Integer))
  let y = 2.5 :: Double
  print (negate y, abs (negate y), signum (negate y), signum (0 :: Double), -y * 2)
  let z = 7 :: Integer
  print (-z, map (\x -> -x) [2 ^ (70 :: Int), 3 :: Integer])
  print (sum (map (\x -> -x) [1.5, 2.5 :: Double]))
