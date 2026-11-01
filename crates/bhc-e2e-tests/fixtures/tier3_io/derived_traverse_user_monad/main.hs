-- traverse over derived Traversable types at a user-defined monad, whose
-- >>= and pure come from its instances (#53).
{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
data Tree a = Leaf | Node (Tree a) a (Tree a) deriving (Show, Functor, Foldable, Traversable)

newtype Counter a = Counter { runCounter :: Int -> (a, Int) }
instance Functor Counter where
  fmap f (Counter g) = Counter (\s -> let (a, s') = g s in (f a, s'))
instance Applicative Counter where
  pure a = Counter (\s -> (a, s))
  Counter f <*> Counter g = Counter (\s -> let (h, s1) = f s; (a, s2) = g s1 in (h a, s2))
instance Monad Counter where
  Counter g >>= k = Counter (\s -> let (a, s') = g s in runCounter (k a) s')

tick :: Int -> Counter (Int, Int)
tick x = Counter (\s -> ((s, x), s + 1))

main :: IO ()
main = do
  let t = Node (Node Leaf 10 Leaf) (20 :: Int) (Node Leaf 30 Leaf)
  print (fst (runCounter (traverse tick t) 0))
  print (fst (runCounter (traverse tick (Just 5)) 7))
