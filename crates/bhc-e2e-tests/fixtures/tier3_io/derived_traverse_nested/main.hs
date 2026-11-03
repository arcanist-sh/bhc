-- traverse over derived Traversable types with recursive and nested fields,
-- at Maybe, Either, IO and the list monad; sequenceA and traverse over Maybe (#53).
{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
data Tree a = Leaf | Node (Tree a) a (Tree a) deriving (Show, Functor, Foldable, Traversable)
data Rose a = Rose a [Rose a] deriving (Show, Functor, Foldable, Traversable)
data R a = R [a] (Maybe a) Int deriving (Show, Functor, Foldable, Traversable)
data P a = P a Int a deriving (Show, Functor, Foldable, Traversable)

half :: Int -> Maybe Int
half n = if even n then Just (n `div` 2) else Nothing

check :: Int -> Either String Int
check n = if n > 0 then Right n else Left ("bad " ++ show n)

main :: IO ()
main = do
  let t = Node (Node Leaf 2 Leaf) (4 :: Int) (Node Leaf 6 Leaf)
  print (traverse Just t)
  print (traverse half t)
  print (traverse half (Node Leaf 3 Leaf))
  print (traverse check t)
  print (traverse check (Node Leaf 1 (Node Leaf (-5) Leaf)))
  t' <- traverse (\x -> print x >> return (x * 10)) t
  print t'
  print (traverse half (Rose 2 [Rose 4 [], Rose 8 [Rose 10 []]]))
  print (traverse half (R [2, 4] (Just 6) 7))
  print (traverse half (R [2, 3] Nothing 7))
  print (traverse (\x -> [x, x + 1]) (P 1 0 (10 :: Int)))
  print (traverse (\x -> [x, x * 2]) [1, 10 :: Int])
  print (sequenceA [Just 1, Just (2 :: Int)])
  print (sequenceA [Just 1, Nothing :: Maybe Int])
  print (sequenceA [[1, 2], [3 :: Int]])
  print (traverse half (Just 8))
  r <- mapM (\x -> return (x + 1)) (P 1 2 (3 :: Int))
  print r
