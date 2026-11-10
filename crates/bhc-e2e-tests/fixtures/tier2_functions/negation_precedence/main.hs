-- Prefix negation has precedence 6: its operand takes the operators that bind
-- tighter. `-7 `mod` 3` is `-(7 `mod` 3)`, `-2 ^ 2` is `-4`. bhc negated the
-- first term only.
f :: Int -> Int
f x = x + 1

main :: IO ()
main = do
  print (-7 `mod` 3 :: Int, negate 7 `mod` 3 :: Int, (-7) `mod` 3 :: Int)
  print (-2 ^ 2 :: Int, (-2) ^ 2 :: Int, -2 * 3 + 1 :: Int)
  print (- f 3, -1 + 2 :: Int, [-1, -2 :: Int])
  print (-x `div` 2, -3.5 :: Double)
  where
    x = 7 :: Int
