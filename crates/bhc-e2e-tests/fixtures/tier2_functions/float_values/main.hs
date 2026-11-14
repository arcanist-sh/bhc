-- Float values through functions and constructor fields, and shown inside
-- lists, tuples, Maybe and derived Show (#156).
module Main where

data V = V Float Float
  deriving (Show, Eq)


getX :: V -> Float
getX (V x _) = x

half :: Float -> Float
half x = x / 2

main :: IO ()
main = do
  let a = 2.5 :: Float
  print a
  print (a + 1, a * 2, a / 2)
  print [2.5 :: Float, 1.5, -0.25]
  print (Just (2.5 :: Float))
  print (2.5 :: Float, 'a')
  print (map (+ 1) [1.5 :: Float])
  print (sum [1.5, 2.5 :: Float], maximum [1.5, 2.5 :: Float])
  print (Left 0.5 :: Either Float Int)
  print [(1 :: Int, 0.1 :: Float)]
  print (V 1.5 (-2.5))
  print (0 :: Float, [0 :: Float], 1.0e-3 :: Float, 1.0e10 :: Float)
  print (zip [1.5 :: Float] [2.5 :: Double])
  print (1.0e-3 :: Float, 1.5e-4 :: Float, 123456789 :: Float, -2.5e-5 :: Float)
  print (1 / 3 :: Float, 1.0e7 :: Float, 0.1 :: Float, 16777216 :: Float)
  print [1 / 0 :: Float, -1 / 0]
  let v = V 1.5 (-2.5)
  print (getX v, getX v + 1, half 5, half (getX v))
