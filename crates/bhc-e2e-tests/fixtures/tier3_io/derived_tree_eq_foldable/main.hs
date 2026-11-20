-- Derived Eq/Ord on a recursive parameterized type, and Foldable consumers
-- over its derived instance (#104).
{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
module Main where

import Data.Foldable (foldl', for_, toList, traverse_)

data Tree a = Leaf | Node (Tree a) a (Tree a)
  deriving (Show, Eq, Ord, Functor, Foldable, Traversable)

t :: Tree Int
t = Node (Node Leaf 1 Leaf) 2 (Node Leaf 3 Leaf)

build :: Int -> Tree Int
build n = Node Leaf n Leaf

main :: IO ()
main = do
  -- derived Eq and Ord recurse into the subtrees
  print (fmap (* 2) t == fmap (+ 0) (fmap (* 2) t))
  print (Node Leaf (1 :: Int) Leaf == Node Leaf 1 Leaf, build 3 == build 3)
  print (build (1 + 2) == Node Leaf 3 Leaf, build 3 == build 4, t == Leaf)
  print (compare t (fmap (+ 1) t), t < t, build 1 < build 2, Leaf < t)
  print (fmap (* 2) t)
  -- Foldable consumers
  print (sum t, product t, length t, maximum t, minimum t)
  print (toList t, foldr (:) [] t)
  print (foldl' (+) 0 t, foldl (flip (:)) [] t)
  print (foldMap (\x -> [x, x]) t)
  print (elem 2 t, any even t, all even t, null t, null (Leaf :: Tree Int))
  mapM_ print t
  for_ t print
  traverse_ print t
  -- Traversable
  r <- traverse (\x -> pure (x + 1)) t
  print r
  mapM_ print =<< traverse (\x -> pure (x + 1)) (Node Leaf (1 :: Int) Leaf)
  pure (build 7) >>= mapM_ print
  let go = mapM_ print :: Tree Int -> IO ()
  go (build 9)
  print (traverse (\x -> if x > 0 then Just x else Nothing) t)
  print (traverse (\x -> if x > 1 then Just x else Nothing) t)
  -- Data.Foldable's names on plain lists
  for_ [10, 20 :: Int] print
  traverse_ print [30 :: Int]
  print (foldl' (+) 0 [1, 2, 3 :: Int], foldMap (\x -> [x, x]) [1, 2 :: Int])
