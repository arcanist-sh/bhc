-- ADTs, pattern matching, recursion, user-defined typeclass.
-- Compiles end-to-end:
--
--   bhc demos/02-types.hs -o /tmp/types
--   /tmp/types
--
-- Output:
--   == Trees ==
--   Leaf
--   Branch 2 (children: Branch 1, Branch 3)
--   == Folds ==
--   sum = 6, depth = 2, count = 3
module Main where

data Tree = Leaf | Branch Tree Int Tree

class Summary a where
  describe :: a -> String

valueOrDash :: Tree -> String
valueOrDash Leaf = "Leaf"
valueOrDash (Branch _ v _) = "Branch " ++ show v

instance Summary Tree where
  describe Leaf = "Leaf"
  describe (Branch l v r) =
    "Branch " ++ show v
      ++ " (children: " ++ valueOrDash l ++ ", " ++ valueOrDash r ++ ")"

sumTree :: Tree -> Int
sumTree Leaf = 0
sumTree (Branch l v r) = sumTree l + v + sumTree r

depthTree :: Tree -> Int
depthTree Leaf = 0
depthTree (Branch l _ r) =
  let dl = depthTree l
      dr = depthTree r
  in 1 + (if dl >= dr then dl else dr)

countBranches :: Tree -> Int
countBranches Leaf = 0
countBranches (Branch l _ r) = 1 + countBranches l + countBranches r

-- A small fixed tree:    2
--                       / \
--                      1   3
sampleTree :: Tree
sampleTree = Branch (Branch Leaf 1 Leaf) 2 (Branch Leaf 3 Leaf)

main :: IO ()
main = do
  putStrLn "== Trees =="
  putStrLn (describe Leaf)
  putStrLn (describe sampleTree)
  putStrLn "== Folds =="
  putStrLn ("sum = " ++ show (sumTree sampleTree)
         ++ ", depth = " ++ show (depthTree sampleTree)
         ++ ", count = " ++ show (countBranches sampleTree))
