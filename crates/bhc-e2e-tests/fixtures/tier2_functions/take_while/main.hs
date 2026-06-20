small :: Int -> Bool
small n = n < 4

lows :: [Int]
lows = takeWhile small [1, 2, 3, 4, 5, 1]

highs :: [Int]
highs = dropWhile small [1, 2, 3, 4, 5, 1]

main :: IO ()
main = do
  print lows
  print highs
