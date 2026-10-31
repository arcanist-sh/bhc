-- Integer literals beyond 64 bits, Integer literal patterns, and
-- negate/abs/signum at Integer (#57).
big :: Integer -> String
big 18446744073709551616 = "two to the 64"
big 0 = "zero"
big _ = "other"

small :: Integer -> String
small 5 = "five"
small n = "not five: " ++ show n

kind :: Integer -> String
kind n = case n of
  7 -> "seven"
  -1 -> "minus one"
  _ -> "nope"

main :: IO ()
main = do
  print (18446744073709551621 :: Integer)
  print [18446744073709551621, -170141183460469231731687303715884105727 :: Integer]
  print (take 2 [18446744073709551616, 18446744073709551621 :: Integer ..])
  print (18446744073709551616 * 2 :: Integer)
  print (0xFFFFFFFFFFFFFFFFFF :: Integer)
  mapM_ (putStrLn . big) [18446744073709551616, 0, 3]
  putStrLn (small 5)
  putStrLn (small 6)
  mapM_ (putStrLn . kind) [7, -1, 8]
  print (maxBound :: Int, 42 :: Int)
  print (abs (-5 :: Integer), signum (-5 :: Integer), negate (3 :: Integer))
  print (abs (-5 :: Int), signum (-5 :: Int), negate (3 :: Int))
  print (map negate [1, 2 :: Integer], map abs [-1, 2 :: Integer])
