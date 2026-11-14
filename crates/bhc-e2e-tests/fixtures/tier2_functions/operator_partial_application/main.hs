-- An operator applied in prefix form to fewer arguments than it takes:
-- `(+) 1`, `(==) k`. The saturated application and the bare operator
-- worked; this aborted with "stub: + not implemented".
import Data.List (sortBy)

applyAll :: [Int -> Int] -> Int -> [Int]
applyAll fs x = map (\f -> f x) fs

main :: IO ()
main = do
  print (map ((+) 1) [1, 2, 3 :: Int])
  print (map (\f -> f 10) (map (\k -> (+) k) [1, 2 :: Int]))
  print (foldr (\k -> (+) k) 0 [1 .. 10 :: Int])
  print (let inc = (+) (1 :: Int) in inc 5)
  print (zipWith (\a -> (*) a) [1, 2] [3, 4 :: Int])
  print (applyAll [(+) 1, (*) 3, (-) 10, subtract 1] 2)
  print (filter ((==) 3) [1, 3, 2, 3 :: Int], filter ((/=) 3) [1, 3, 2 :: Int])
  print (takeWhile ((>) 5) [1, 3, 7, 2 :: Int], any ((<) 3) [1, 2, 4 :: Int], all ((>=) 9) [1, 2 :: Int])
  print (map ((+) 1.5) [1, 2 :: Double], map ((*) 2) [3, 4 :: Integer], map ((/) 10) [2, 4 :: Double])
  print (map ((-) 100) [1 :: Integer, 2 ^ 70])
  print (map ((==) 'a') "abca", map ((==) "b") ["a", "b"])
  print (map (div 100) [3, 7 :: Int], map (mod 100) [3, 7 :: Int])
  -- The saturated and the bare forms are unchanged.
  print ((+) 1 (2 :: Int), foldr (+) 0 [1, 2, 3 :: Int], map (uncurry (*)) [(2, 3 :: Int)], (3 :: Int) + 4)
