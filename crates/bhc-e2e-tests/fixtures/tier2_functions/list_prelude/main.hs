isEven :: Int -> Bool
isEven n = mod n 2 == 0

threes :: [Int]
threes = take 3 [10, 20, 30, 40, 50]

dropped :: [Int]
dropped = drop 2 [1, 2, 3, 4, 5]

reps :: [Int]
reps = replicate 4 7

sums :: [Int]
sums = zipWith (+) [1, 2, 3] [10, 20, 30]

main :: IO ()
main = do
  print threes
  print dropped
  print reps
  print sums
  print (null ([] :: [Int]))
  print (null [1 :: Int])
  print (head [9, 8, 7 :: Int])
  print (product [1, 2, 3, 4 :: Int])
  print (all isEven [2, 4, 6])
  print (any isEven [1, 3, 4])
  print (and [True, True, False])
  print (or [False, False, True])
