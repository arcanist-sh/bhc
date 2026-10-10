-- `f =<< m` IS `m >>= f`, so at a user monad the flipped spelling must reach
-- the same instance. It used not to reach the dispatch at all, and the generic
-- builtin bind ran the action as if it were IO.
module Main where

data Logged a = Logged String a

instance Functor Logged where
  fmap f (Logged s a) = Logged s (f a)

instance Applicative Logged where
  pure = Logged ""
  Logged s f <*> Logged t a = Logged (s ++ t) (f a)

instance Monad Logged where
  Logged s a >>= k = case k a of
    Logged t b -> Logged (s ++ "." ++ t) b

step :: Int -> Logged Int
step n = Logged ("step" ++ show n) (n + 1)

main :: IO ()
main = case step =<< step 1 of
  Logged s n -> putStrLn (s ++ " " ++ show n)
