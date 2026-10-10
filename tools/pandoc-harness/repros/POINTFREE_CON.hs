-- `return $ return x` has two DIFFERENT monads in it, and `$` must not hide
-- either one: the outer `return` belongs to the result type's monad, the inner
-- to the one inside it. pandoc's markdown reader ends nearly every inline
-- parser this way (`return $ return $ B.str t`, parser outside, Future
-- inside), and taking the parser's `return` for both made the parser's action
-- the value itself.
module Main where

data Outer a = Outer String a

data Inner a = Inner String a

instance Functor Outer where
  fmap f (Outer s a) = Outer s (f a)

instance Applicative Outer where
  pure = Outer "o"
  Outer s f <*> Outer t a = Outer (s ++ t) (f a)

instance Monad Outer where
  Outer s a >>= k = case k a of
    Outer t b -> Outer (s ++ t) b

instance Functor Inner where
  fmap f (Inner s a) = Inner s (f a)

instance Applicative Inner where
  pure = Inner "i"
  Inner s f <*> Inner t a = Inner (s ++ t) (f a)

instance Monad Inner where
  Inner s a >>= k = case k a of
    Inner t b -> Inner (s ++ t) b

dollars :: Outer (Inner Int)
dollars = return $ return 5

parens :: Outer (Inner Int)
parens = return (return 7)

label :: Outer (Inner Int) -> String
label (Outer s (Inner t n)) = s ++ t ++ show n

main :: IO ()
main = do
  putStrLn (label dollars)
  putStrLn (label parens)
