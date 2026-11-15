-- Arithmetic at a type the program gives a Num instance, and the folds at
-- the types that have one, stay accepted: the check that rejects
-- `sum "abc"` looks for a declared instance first.
newtype V = V Int deriving (Show)
instance Num V where
  V a + V b = V (a + b)
  V a * V b = V (a * b)
  negate (V a) = V (negate a)
  abs (V a) = V (abs a)
  signum (V a) = V (signum a)
  fromInteger n = V (fromInteger n)

instance (Num a, Num b) => Num (a, b) where
  (a, b) + (c, d) = (a + c, b + d)
  (a, b) * (c, d) = (a * c, b * d)
  negate (a, b) = (negate a, negate b)
  abs (a, b) = (abs a, abs b)
  signum (a, b) = (signum a, signum b)
  fromInteger n = (fromInteger n, fromInteger n)

main :: IO ()
main = do
  print (V 1 + V 2)
  print (sum [1.5, 2.5 :: Double], product [1, 2, 3 :: Integer])
  print (maximum "hello", minimum [3, 1, 2 :: Int])
  print (fromIntegral (length "abc") / 2 :: Double)
