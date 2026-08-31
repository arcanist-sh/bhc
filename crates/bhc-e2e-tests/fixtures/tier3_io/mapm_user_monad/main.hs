module Main where

-- `mapM_`/`mapM` at a monad whose actions are VALUES, not something running
-- as it is built. Lowering them to `map` only works for bhc's eager IO.
newtype Counter a = Counter { runCounter :: Int -> (a, Int) }

instance Functor Counter where
  fmap f (Counter g) = Counter (\s -> let (a, s') = g s in (f a, s'))

instance Applicative Counter where
  pure a = Counter (\s -> (a, s))
  Counter f <*> Counter g =
    Counter (\s -> let (h, s1) = f s
                       (a, s2) = g s1
                   in (h a, s2))

instance Monad Counter where
  Counter g >>= k = Counter (\s -> let (a, s1) = g s in runCounter (k a) s1)

tick :: Int -> Counter ()
tick n = Counter (\s -> ((), s + n))

double :: Int -> Counter Int
double n = Counter (\s -> (n * 2, s + 1))

main :: IO ()
main = do
  print (snd (runCounter (mapM_ tick [1, 2, 3, 4]) 0))
  print (fst (runCounter (mapM double [1, 2, 3]) 0))
  print (snd (runCounter (mapM double [1, 2, 3]) 0))
