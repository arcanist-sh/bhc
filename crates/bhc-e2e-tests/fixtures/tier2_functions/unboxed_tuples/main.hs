-- UnboxedTuples, read as boxed tuples (#36).
{-# LANGUAGE UnboxedTuples #-}
divMod' :: Int -> Int -> (# Int, Int #)
divMod' a b = (# a `div` b, a `mod` b #)

single :: Int -> (# Int #)
single x = (# x + 1 #)

main :: IO ()
main = do
  case divMod' 17 5 of
    (# q, r #) -> print (q, r)
  case single 4 of
    (# y #) -> print y
