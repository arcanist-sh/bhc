-- `quot`, `rem`, `quotRem`, `divMod`, `gcd` and `lcm` at `Integer` (#78). They were
-- pinned to `Int`, so an `Integer` use ran machine arithmetic on bignum pointers,
-- or built a negated literal operand as a machine integer.
main :: IO ()
main = do
  print ((-7) `quot` 2 :: Integer, 2 :: Integer)
  print ((-7) `rem` 2 :: Integer, 2 :: Integer)
  print ((-7) `div` 2 :: Integer, (-7) `mod` 2 :: Integer)
  print ((-7) `quot` 2 :: Int, (-7) `rem` 2 :: Int)
  print (quotRem (-7) 2 :: (Int, Int), divMod (-7) 2 :: (Int, Int))
  print (quotRem 7 (-2) :: (Int, Int), divMod 7 (-2) :: (Int, Int))
  print (quotRem (-7) 2 :: (Integer, Integer), divMod (-7) 2 :: (Integer, Integer))
  let big = 2 ^ 70 :: Integer
  print (big `quot` 3, big `rem` 3)
  print (quotRem big 7, divMod (negate big) 7)
  print (gcd 12 18 :: Integer, lcm 4 6 :: Integer)
  print (gcd big (2 ^ 65), lcm big 3)
  print (gcd (-12) 18 :: Int, lcm (-4) 6 :: Int)
  print (foldr gcd 0 [12, 18, 24 :: Int])
