{-# LANGUAGE DeriveFoldable #-}
-- Foldable consumers over containers that are not lists (#63): a derived
-- recursive instance, a hand-written instance, a derived newtype over a
-- container (pandoc's `Many a = Many (Seq a)`) and Maybe. `sum` etc. were
-- typed at lists only, and the derived `foldr` applied its function to
-- whole subtrees instead of folding into them.
import Control.Monad (forM_)
import Data.Foldable (toList)
import qualified Data.Sequence as Seq

data Tree a = Leaf | Node (Tree a) a (Tree a) deriving Foldable

data Pair a = Pair a a

newtype Bag a = Bag [a] deriving Foldable

newtype SBag a = SBag (Seq.Seq a) deriving Foldable

instance Foldable Pair where
  foldr f z (Pair x y) = f x (f y z)

t :: Tree Int
t = Node (Node Leaf 1 Leaf) 2 (Node Leaf 3 Leaf)

main :: IO ()
main = do
  print (sum t, product t, length t)
  print (null t, null (Leaf :: Tree Int))
  print (maximum t, minimum t)
  print (elem 2 t, elem 7 t, notElem 7 t)
  print (any (> 2) t, all (> 2) t)
  print (toList t)
  print (toList (Node Leaf 'a' (Node Leaf 'b' Leaf)))
  print (sum (Node Leaf (1 :: Int) Leaf))
  mapM_ print (Node Leaf 'a' (Node Leaf 'b' Leaf))
  forM_ t print
  print (sum (Pair 4 5 :: Pair Int), length (Pair 'a' 'b'))
  print (sum (Bag [1, 2, 3 :: Int]), length (Bag "ab"), null (Bag ""), toList (Bag "xy"))
  print (foldr (:) [] (Bag "abc"), foldr (:) [] (SBag (Seq.fromList "xyz")))
  print (sum (Just 3 :: Maybe Int), length (Nothing :: Maybe Int))
  mapM_ print (Just 'z')
  print (sum [1, 2, 3 :: Int], length "abc", elem 'b' "abc")
