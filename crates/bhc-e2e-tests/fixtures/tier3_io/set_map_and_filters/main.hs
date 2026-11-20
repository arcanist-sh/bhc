-- `Data.Set` functions at their real types (`S.map` with a section), and
-- Set/Map filters whose predicate is a comparison (#102).
module Main where

import qualified Data.Map as M
import qualified Data.Set as S
import Data.Set (Set)

evens :: Set Int -> Set Int
evens = S.filter even

main :: IO ()
main = do
  let s = S.fromList [1 .. 10 :: Int]
  print (S.toList (S.map (`mod` 3) s))
  print (S.toList (S.map (\x -> x `mod` 3) s))
  print (S.toList (S.map (+ 1) (S.fromList [1 .. 3 :: Int])))
  print (S.toList (S.map (* 2) (S.fromList [1 .. 3 :: Int])))
  print (S.toList (S.map (subtract 1) (S.fromList [1 .. 3 :: Int])))
  print (S.toList (S.filter (> 7) s))
  print (S.toList (evens s))
  print (S.toList (S.insert 0 (S.delete 10 s)))
  print (S.toList (S.union (S.fromList [1, 2 :: Int]) (S.fromList [2, 3])))
  print (S.toList (S.intersection s (S.fromList [9, 10, 11])))
  print (S.toList (S.difference s (S.fromList [2 .. 9])))
  print (S.isSubsetOf (S.fromList [1, 2 :: Int]) s, S.isSubsetOf s (S.fromList [1 :: Int]))
  print (S.toList (S.deleteMin s) == [2 .. 10], S.toList (S.deleteMax s) == [1 .. 9])
  print (S.size (S.map (`div` 4) s), S.member 2 (S.map (`div` 4) s))
  print (S.null (S.empty :: Set Int), S.size (S.insert 'x' S.empty))
  print (S.toList (foldr S.insert S.empty [3, 1, 2 :: Int]))
  let (small, big) = S.partition (< 4) s
  print (S.toList small, S.size big)
  let m = M.fromList (zip [1 .. 6 :: Int] "abcdef")
  print (M.toList (M.filter (> 'c') m))
  print (M.toList (M.filterWithKey (\k _ -> k `mod` 2 == 0) m))
  print (M.toList (M.filter (`elem` "ace") m))
