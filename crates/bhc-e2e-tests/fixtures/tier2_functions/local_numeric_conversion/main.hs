-- A local value binding without a signature that converts a number keeps
-- the type its use gives it (the monomorphism restriction): `where w =
-- fromInteger n` used as an Int. Generalized, the conversion compiled as the
-- identity and `w + 1` added 1 to the Integer's address. A machine integer
-- bound as a Double (`where x = fromIntegral k`, with or without a signature)
-- kept its integer bits, so `x / 4` printed 1.0e-323.
c1 :: Integer -> Int
c1 n = w + 1
  where
    w = fromInteger n

c2 :: Integer -> Int
c2 n = let w = fromIntegral n in w * 2

c3 :: Double -> Int
c3 d = r + t
  where
    r = round d
    t = truncate (d * 10)

c4 :: Int -> Double
c4 k = x / 4
  where
    x = fromIntegral k

c5 :: Int -> Double
c5 k = x
  where
    x :: Double
    x = fromIntegral k

c6 :: Int -> Double
c6 k = let x = fromIntegral (k + 1) :: Double in x * 2

main :: IO ()
main = do
  print (c1 41)
  print (c2 21)
  print (c3 2.6)
  print (c4 10)
  print (c5 10)
  print (c6 10)
