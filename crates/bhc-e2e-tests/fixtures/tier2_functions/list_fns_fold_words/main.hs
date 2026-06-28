-- Data.List.foldl' and the unwords/unlines/zipWith3 prelude functions, all
-- previously missing from the WASM backend's synthesized prelude.
import Data.List (foldl')

main :: IO ()
main = do
    print (foldl' (+) 0 [1 .. 100 :: Int])
    print (foldl' max 0 [3, 7, 2, 9, 1 :: Int])
    putStrLn (unwords ["the", "quick", "brown", "fox"])
    putStr (unlines ["line1", "line2", "line3"])
    print (zipWith3 (\a b c -> a + b + c) [1, 2, 3] [10, 20, 30] [100, 200, 300 :: Int])
