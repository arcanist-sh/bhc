module Main where

main :: IO ()
main = do
    putStrLn (show (sortOn (\x -> 0 - x) [3,1,4,1,5]))
    putStrLn (show (nubBy (\a b -> rem a 3 == rem b 3) [1,2,4,5,7]))
    putStrLn (show [length (groupBy (\a b -> rem a 2 == rem b 2) [1,3,2,4,1])])
    putStrLn (show (deleteBy (\a b -> rem a 10 == rem b 10) 3 [11,12,13,14]))
    putStrLn (show (unionBy (\a b -> rem a 10 == rem b 10) [1,2] [12,3]))
    putStrLn (show (intersectBy (\a b -> rem a 10 == rem b 10) [1,2,3] [11,13]))
    putStrLn (show (insert 3 [1,2,4,5]))
    putStrLn (show (concat (maybeToList (stripPrefix [1,2] [1,2,3,4]))))
    putStrLn (show (concat (maybeToList (stripPrefix [1,3] [1,2,3,4]))))
    putStrLn (show (take 100 (snd (mapAccumL (\a x -> (a + x, a + x)) 0 [1,2,3]))))
    putStrLn (show [fst (mapAccumL (\a x -> (a + x, a + x)) 0 [1,2,3])])
    putStrLn (show (take 100 (snd (mapAccumR (\a x -> (a + x, a + x)) 0 [1,2,3]))))
    putStrLn (show [fst (mapAccumR (\a x -> (a + x, a + x)) 0 [1,2,3])])
