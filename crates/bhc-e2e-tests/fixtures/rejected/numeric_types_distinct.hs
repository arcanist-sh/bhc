-- Int, Integer and Double are different types. Each of these compiled: the
-- unifier treated every integer type as one type and every floating type
-- as another, and `fromIntegral` accepted a Double (#97). The first
-- crashed at run time, the others printed a wrong result.
widen :: Int -> Integer
widen x = x

truncateIt :: Double -> Int
truncateIt d = d

main :: IO ()
main = do
  print (widen 3)
  print (truncateIt 1.5)
  print ((1 :: Int) + (2 :: Integer))
  print (fromIntegral (1.5 :: Double) :: Int)
