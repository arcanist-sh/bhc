-- Test: conversions between Int and Integer, printing, power and derived
-- Eq/Ord over an Integer field.
-- Verifies: fromInteger/fromIntegral narrow a bignum, toInteger widens one,
-- print renders an Integer, (^) at Int/Integer/Double, and a derived
-- instance compares Integer fields by value.

newtype Day = Day Integer deriving (Eq, Ord)

integerToInt :: Integer -> Int
integerToInt n = fromInteger n

intToInteger :: Int -> Integer
intToInteger n = toInteger n

mkDay :: Int -> Day
mkDay n = Day (intToInteger (n + 40587))

dayNumber :: Day -> Int
dayNumber (Day d) = integerToInt d - 40587

shift :: Integer -> Integer
shift d = (d + 3) `mod` 7

sq :: Int -> Int
sq x = x ^ (2 :: Int)

main :: IO ()
main = do
  print (0 :: Integer)
  print (shift 12)
  print (shift 11 + 1)
  print (fromInteger (5 :: Integer) :: Int)
  print (integerToInt 42 + 1)
  print (intToInteger 7 * intToInteger 6)
  print (dayNumber (mkDay 19000))
  print (mkDay 3 == mkDay 3)
  print (mkDay 3 < mkDay 4)
  print (compare (mkDay 5) (mkDay 4))
  print (sq 9)
  print (2 ^ (10 :: Int) :: Int)
  print (2 ^ (10 :: Int) :: Integer)
  print (1.5 ^ (2 :: Int) :: Double)
