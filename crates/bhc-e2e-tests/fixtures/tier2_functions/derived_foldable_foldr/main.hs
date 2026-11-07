{-# LANGUAGE DeriveFoldable #-}
-- `foldr` called directly on a type with only a derived `Foldable` instance.
-- Held in a variable, the container was folded as a list, forcing its raw
-- elements (#87); and the derived `foldr` passed its accumulator to `f` as a
-- thunk, so an `Int` result printed as an address (#64).
data Tree a = Leaf | Node (Tree a) a (Tree a) deriving (Foldable)

data Rose a = Rose a [Rose a] deriving (Foldable)

tree :: Tree Int
tree = Node (Node Leaf 1 Leaf) 5 (Node Leaf 7 (Node Leaf 9 Leaf))

rose :: Rose Int
rose = Rose 1 [Rose 2 [Rose 4 []], Rose 3 []]

main :: IO ()
main = do
  print (foldr (:) [] tree)
  print (foldr (+) 0 tree)
  print (foldr (\x acc -> acc * 2 + x) 1 tree)
  print (foldr (\x acc -> fromIntegral x + acc) 0.5 tree :: Double)
  print (foldr max 0 tree)
  print (foldr (+) 0 (Node Leaf 3 (Node Leaf 4 Leaf) :: Tree Int))
  print (foldr (:) [] rose)
  print (foldr (+) 0 rose)
  print (sum tree, product tree, length tree, maximum tree, elem 7 tree)
  print (sum rose, length rose)
