-- A polymorphic recursive fold builds its accumulator with an unknown
-- function: `f x (foldT f z r)`. That argument calls `foldT` again, and was
-- passed as a thunk, which at `Int` or `Double` was read as a number and
-- printed as an address (#64).
data Tree a = Leaf | Node (Tree a) a (Tree a)

foldT :: (a -> b -> b) -> b -> Tree a -> b
foldT _ z Leaf = z
foldT f z (Node l x r) = foldT f (f x (foldT f z r)) l

tree :: Tree Int
tree = Node (Node Leaf 1 Leaf) 5 (Node Leaf 7 (Node Leaf 9 Leaf))

main :: IO ()
main = do
  print (foldT (+) 0 (Leaf :: Tree Int))
  print (foldT (+) 0 (Node Leaf 5 Leaf :: Tree Int))
  print (foldT (+) 0 tree)
  print (foldT (*) 1 tree)
  print (foldT (\x acc -> acc + fromIntegral x) 0.5 tree :: Double)
  print (foldT max 0 tree)
  print (foldT (:) [] tree)
  putStrLn (foldT (\x s -> show x ++ s) "" tree)
