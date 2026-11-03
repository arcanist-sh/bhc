-- Class-polymorphic functions (Num, Eq, Ord) used at Int, Integer, Double and
-- String: each concrete use gets a specialized copy (#57).
sq :: Num a => a -> a
sq x = x * x

sumSq :: Num a => [a] -> a
sumSq = foldr (\x acc -> x * x + acc) 0

twice :: Num a => a -> a
twice x = x + x + fromInteger 0

biggest :: Ord a => a -> a -> a
biggest a b = if a > b then a else b

same :: Eq a => a -> a -> Bool
same a b = a == b

main :: IO ()
main = do
  print (sq (3 :: Int), sq (2 ^ 40 :: Integer), sq (1.5 :: Double))
  print (map sq [1, 2, 3 :: Integer])
  print (sumSq [1, 2, 3 :: Integer], sumSq [0.5, 1.5 :: Double])
  print (twice (2 ^ 63 :: Integer))
  print (biggest (2 ^ 70) (3 :: Integer), biggest 'a' 'b')
  print (same (2 ^ 70) (2 ^ 70 :: Integer), same "ab" "ab")
