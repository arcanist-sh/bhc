module Main where

newtype F s a = F { runF :: s -> a }

instance Functor (F s) where
  fmap f (F g) = F (f . g)
instance Applicative (F s) where
  pure x = F (const x)
  F f <*> F g = F (\s -> f s (g s))
instance Monad (F s) where
  F g >>= k = F (\s -> runF (k (g s)) s)

instance Semigroup a => Semigroup (F s a) where
  (<>) = liftA2 (<>)
instance (Semigroup a, Monoid a) => Monoid (F s a) where
  mempty = return mempty
  mappend = (<>)

cat :: Monoid m => [m] -> m
cat = foldr (<>) mempty

main :: IO ()
main = do
  print (sum (runF (F (\s -> [s, s])) (7 :: Int)))
  print (length (runF (mempty :: F Int [Int]) 0))
  print (sum (runF (F (\s -> [s]) <> F (\s -> [s + 1])) 5))
  print (sum (runF (mconcat [F (\s -> [s]), F (\s -> [s * 2])]) 3))
  print (sum (runF (cat [F (\s -> [s]), mempty, F (\s -> [s * 3])]) 2))
