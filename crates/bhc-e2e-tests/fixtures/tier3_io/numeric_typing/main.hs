-- Test: numeric values keep their types through constructors, records,
-- tuples and conversions.
-- Verifies: an integer literal in an Integer constructor field is a bignum;
-- a record selector keeps its field type; a tuple component compared with a
-- literal in a guard is an Integer; fromIntegral reaches Double from Int and
-- Integer; msum over a mapped list of Maybes.

import Control.Monad (msum)

newtype Day = ModifiedJulianDay {toModifiedJulianDay :: Integer}

data Rec = Rec {recCount :: Integer, recName :: String}

addDays :: Integer -> Day -> Day
addDays n (ModifiedJulianDay d) = ModifiedJulianDay (d + n)

diffDays :: Day -> Day -> Integer
diffDays (ModifiedJulianDay a) (ModifiedJulianDay b) = a - b

triple :: Int -> (Integer, Int, Int)
triple n = (toInteger n, 1, 2)

inRange :: Int -> Bool
inRange n = case triple n of
  (y, _, _) | y >= 1601 && y <= 9999 -> True
  _ -> False

split :: Double -> (Int, Int, Double)
split secs =
  let whole = floor secs :: Int
      h = whole `div` 3600
      m = (whole `mod` 3600) `div` 60
   in (h, m, secs - fromIntegral (h * 3600 + m * 60))

pick :: Int -> Maybe Int
pick x = if x > 1 then Just x else Nothing

main :: IO ()
main = do
  print (toModifiedJulianDay (ModifiedJulianDay 0))
  print (toModifiedJulianDay (addDays 3 (ModifiedJulianDay 1)))
  print (diffDays (ModifiedJulianDay 10) (ModifiedJulianDay 3))
  print (recCount (Rec 5 "r"))
  print (inRange 2024)
  print (inRange 1500)
  let (h, m, s) = split 80000
  print h
  print m
  print s
  print (fromIntegral (10 ^ (12 :: Int) :: Integer) / 4 :: Double)
  print (msum (map pick [1, 2, 3]))
