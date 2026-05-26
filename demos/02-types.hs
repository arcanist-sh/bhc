-- ADTs, pattern matching, recursion, type classes with dictionary passing.
-- Compiles end-to-end:
--
--   bhc demos/02-types.hs -o /tmp/types
--   /tmp/types
--
-- Output:
--   Leaf
--   Branch (Leaf) 1 (Branch (Leaf) 2 (Leaf))
--   sum = 3, depth = 3, count = 2
module Main where

data Tree a = Leaf | Branch (Tree a) a (Tree a)

insert :: Ord a => a -> Tree a -> Tree a
insert x Leaf = Branch Leaf x Leaf
insert x t@(Branch l v r)
  | x < v     = Branch (insert x l) v r
  | x > v     = Branch l v (insert x r)
  | otherwise = t

class Summary a where
  describe :: a -> String

instance Show a => Summary (Tree a) where
  describe Leaf = "Leaf"
  describe (Branch l v r) =
    "Branch (" ++ describe l ++ ") " ++ show v ++ " (" ++ describe r ++ ")"

sumTree :: Num a => Tree a -> a
sumTree Leaf = 0
sumTree (Branch l v r) = sumTree l + v + sumTree r

depth :: Tree a -> Int
depth Leaf = 0
depth (Branch l _ r) = 1 + max (depth l) (depth r)

countBranches :: Tree a -> Int
countBranches Leaf = 0
countBranches (Branch l _ r) = 1 + countBranches l + countBranches r

main :: IO ()
main = do
  let empty = Leaf :: Tree Int
      t     = foldr insert empty [2, 1]
  putStrLn (describe (Leaf :: Tree Int))
  putStrLn (describe t)
  putStrLn ("sum = " ++ show (sumTree t)
         ++ ", depth = " ++ show (depth t)
         ++ ", count = " ++ show (countBranches t))
