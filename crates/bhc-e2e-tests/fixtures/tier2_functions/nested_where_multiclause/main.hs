-- A function in a `where` (or `let`) block whose own `where` defines a
-- multi-clause helper keeps that helper. pandoc-types' `placeRowSection`
-- has `dropAtMostWhile p n = go 0 where go acc (l:ls) | … ; go acc l = …`;
-- the nested `go` was dropped and called as a stub ("stub: go not
-- implemented") as soon as a table was built.
module Main where

-- Single-clause function with parameters, with a signature.
dropSmall :: [Int] -> (Int, [Int])
dropSmall xs = dropAtMostWhile (== 1) 2 xs
  where
    dropAtMostWhile :: (a -> Bool) -> Int -> [a] -> (Int, [a])
    dropAtMostWhile p n = go 0
      where
        go acc (l:ls) | p l && acc < n = go (acc + 1) ls
        go acc l = (acc, l)

-- Simple (parameterless) binding whose `where` has a multi-clause helper.
lenSum :: [Int] -> Int
lenSum xs = total
  where
    total = count xs + 0
      where
        count [] = 0
        count (_:ys) = 1 + count ys

-- A `let`-bound function with a nested `where`.
firstBig :: [Int] -> Maybe Int
firstBig xs =
  let find' t = go xs
        where
          go [] = Nothing
          go (y:ys) | y > t = Just y
                    | otherwise = go ys
  in find' 3

main :: IO ()
main = do
  let (n, rest) = dropSmall [1, 1, 1, 5]
  print n
  print rest
  print (lenSum [4, 5, 6])
  print (firstBig [1, 2, 7, 9])
  print (firstBig [1, 2])
