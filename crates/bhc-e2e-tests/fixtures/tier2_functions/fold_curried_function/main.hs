-- A fold's function need not take both arguments at once. `\k -> M.insert k k`
-- takes the element and returns a function of the accumulator; the fold called
-- its code with two arguments and the second was dropped.
import qualified Data.Map as M
import Data.List (foldl')

ins :: M.Map Int Int -> Int -> M.Map Int Int
ins m k = M.insert k (k * k) m

main :: IO ()
main = do
  -- foldr with a one-parameter lambda returning a function
  print (M.toList (foldr (\k -> M.insertWith (+) (k `mod` 3) k) M.empty [1 .. 10 :: Int]))
  print (M.size (foldr (\k -> M.insert k k) M.empty [1 .. 10 :: Int]))
  print (M.size (foldr (\k -> flip ins k) M.empty [1 .. 10 :: Int]))
  -- foldl and foldl' likewise
  print (M.toList (foldl (\m -> ins m) M.empty [1 .. 4 :: Int]))
  print (M.size (foldl' (\m -> ins m) M.empty [1 .. 10 :: Int]))
  -- The two-parameter forms are unchanged
  print (foldr (\x acc -> x + acc) 0 [1 .. 10 :: Int], foldl (\acc x -> acc * 2 + x) 0 [1, 0, 1 :: Int])
  print (foldr (\x -> \acc -> x : acc) [] "abc", foldl (\acc -> \x -> x : acc) [] "abc")
  print (foldr (\x k -> k . (+ x)) id [1 .. 10 :: Int] 0, M.toList (foldr (flip ins) M.empty [1, 2]) == M.toList (foldl ins M.empty [2, 1]))
