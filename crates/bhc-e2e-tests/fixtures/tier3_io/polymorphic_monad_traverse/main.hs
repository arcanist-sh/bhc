{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
-- Traversals inside `Monad m =>` functions, where the monad is a type
-- variable (#63): `>>=` and `pure` come from the dictionary. These
-- segfaulted; at IO under the closure model the recursive traversal also ran
-- a thunk as an action.
import Data.Traversable (for)

data Tree a = Leaf | Node (Tree a) a (Tree a) deriving (Show, Functor, Foldable, Traversable)

label :: Monad m => (Int -> m Int) -> Maybe Int -> m (Maybe Int)
label g mx = traverse g mx

labelE :: Monad m => (Int -> m Int) -> Either String Int -> m (Either String Int)
labelE g ex = for ex g

labelL :: Monad m => (Int -> m Int) -> [Int] -> m [Int]
labelL g xs = mapM g xs

relabel :: Monad m => (Int -> m Int) -> Tree Int -> m (Tree Int)
relabel g t = traverse g t

main :: IO ()
main = do
  label (\x -> return (x * 3)) (Just 4) >>= print
  label (\x -> return (x * 3)) Nothing >>= print
  print (label (\x -> Just (x * 3)) (Just 4))
  print (label (\_ -> Nothing) (Just 4))
  labelE (\x -> return (x + 1)) (Right 1) >>= print
  labelE (\x -> return (x + 1)) (Left "no") >>= print
  labelL (\x -> return (x + 1)) [1, 2, 3] >>= print
  print (labelL (\x -> if x > 0 then Just x else Nothing) [1, -2])
  let t = Node (Node Leaf 1 Leaf) 2 Leaf
  relabel (\x -> return (x * 10)) t >>= print
  print (relabel (\x -> Just (x + 1)) t)
  print (relabel (\x -> if x > 1 then Nothing else Just x) t)
