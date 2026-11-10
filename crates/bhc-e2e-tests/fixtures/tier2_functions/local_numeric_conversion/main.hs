-- A local value binding without a signature that converts a number keeps
-- the type its use gives it (the monomorphism restriction): `where w =
-- fromInteger n` used as an Int. Generalized, the conversion compiled as the
-- identity and `w + 1` added 1 to the Integer's address.
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

main :: IO ()
main = do
  print (c1 41)
  print (c2 21)
  print (c3 2.6)
