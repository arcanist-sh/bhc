-- A hand-written `Traversable` instance (#66): its `traverse` is checked at
-- `Applicative f => (a -> f b) -> Tree a -> f (Tree b)` and takes that
-- dictionary, and every traversal of the type goes through it, at Maybe,
-- IO and lists, from a `Monad m =>` function too. The instance runs right to
-- left on purpose: it must not be replaced by a derived traversal.
import Data.Traversable (for)

data Tree a = Leaf | Node (Tree a) a (Tree a) deriving Show

instance Functor Tree where
  fmap _ Leaf = Leaf
  fmap f (Node l x r) = Node (fmap f l) (f x) (fmap f r)

instance Foldable Tree where
  foldr _ z Leaf = z
  foldr f z (Node l x r) = foldr f (f x (foldr f z r)) l

instance Traversable Tree where
  traverse _ Leaf = pure Leaf
  traverse f (Node l x r) = flip3 <$> traverse f r <*> f x <*> traverse f l
    where
      flip3 r' x' l' = Node l' x' r'

relabel :: Monad m => (Int -> m Int) -> Tree Int -> m (Tree Int)
relabel g t = traverse g t

-- A recursive `Applicative f =>` function at lists (its `fmap` gets the
-- recursive call unevaluated).
countdown :: Applicative f => Int -> f [Int]
countdown 0 = pure []
countdown n = fmap (n :) (countdown (n - 1))

main :: IO ()
main = do
  let t = Node (Node Leaf 1 Leaf) (2 :: Int) (Node Leaf 3 Leaf)
  print (traverse Just t)
  print (traverse (\x -> if x > 2 then Nothing else Just x) t)
  _ <- traverse print t
  print (sequenceA (fmap Just t))
  r <- mapM (\x -> return (x + 1)) t
  print r
  r2 <- for t (\x -> return (x * 2))
  print r2
  relabel (\x -> return (x * 10)) t >>= print
  print (relabel (\x -> if x > 2 then Nothing else Just x) t)
  print (traverse (\x -> [x, x + 100]) (Node Leaf (1 :: Int) Leaf))
  print (countdown 3 :: [[Int]], countdown 2 :: Maybe [Int])
  print (sum t, length t)
