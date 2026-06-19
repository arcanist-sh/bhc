inc :: Int -> Int
inc x = x + 1

isEven :: Int -> Bool
isEven n = mod n 2 == 0

doubled :: [Int]
doubled = map inc [1, 2, 3]

evens :: [Int]
evens = filter isEven [1, 2, 3, 4, 5, 6]

main :: IO ()
main = do
  print doubled
  print evens
  print (length (map inc [1, 2, 3]))
  print (foldr (+) 0 [1, 2, 3, 4])
  print (foldl (+) 0 [10, 20, 30])
