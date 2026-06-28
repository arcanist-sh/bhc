-- User operators whose names begin with a reserved operator (=>, ->, <-).
-- Maximal munch makes `=>=`, `->-` single operators, not `=>` + `=` etc.
module Main (main, (=>=), (->-)) where

(=>=) :: Int -> Int -> Int
a =>= b = a + b

(->-) :: Int -> Int -> Int
a ->- b = a - b

main :: IO ()
main = do
    print (10 =>= 5)
    print ((10 =>= 5) ->- 3)
