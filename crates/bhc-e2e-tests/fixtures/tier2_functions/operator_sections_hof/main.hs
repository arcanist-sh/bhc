-- Regression: bare operator sections ((+)/(-)/(*)) passed as first-class
-- functions to higher-order functions must each keep their own operation.
-- A native bug keyed every primop wrapper by a name that collapsed all
-- non-alphanumeric operators to the same symbol, so the first section used was
-- reused for all subsequent ones (foldl (+) then foldl (-) both ran +).
main :: IO ()
main = do
    print (foldl (+) 0 [1, 2, 3, 4 :: Int])
    print (foldl (-) 0 [1, 2, 3, 4 :: Int])
    print (foldr (+) 0 [1, 2, 3, 4 :: Int])
    print (foldr (*) 1 [1, 2, 3, 4 :: Int])
    print (zipWith (+) [1, 2, 3] [10, 20, 30 :: Int])
    print (zipWith (*) [1, 2, 3] [10, 20, 30 :: Int])
