{-# LANGUAGE FlexibleInstances #-}
-- A Semigroup/Monoid instance written at a type synonym over a newtype
-- (pandoc-types' `Inlines = Many Inline`), used through `Monoid m =>`
-- functions, the bare `(<>)` operator, and the class defaults.
module Main where

newtype Many a = Many { unMany :: [a] }
type Inl = Many Int

instance Semigroup Inl where
  Many a <> Many b = Many (a ++ b)
instance Monoid Inl where
  mempty = Many mempty
  mappend = (<>)

f :: Monoid m => [m] -> m
f = foldr (<>) mempty

g :: Monoid m => m
g = mempty

h1 :: Monoid m => m -> m -> m
h1 a b = a <> b

h2 :: Semigroup m => m -> m -> m
h2 a b = foldr (<>) b [a]

h3 :: Monoid m => m -> m -> m
h3 a b = foldr (\x y -> x <> y) b [a]

main :: IO ()
main = do
  print (unMany (f [Many [1], Many [2,3]] :: Inl))
  print (unMany (g :: Inl))
  print (unMany (mconcat [Many [4], Many [5]] :: Inl))
  print (length (unMany (mempty :: Inl)))
  print (unMany (h1 (Many [1]) (Many [2]) :: Inl))
  print (unMany (h2 (Many [1]) (Many [2]) :: Inl))
  print (unMany (h3 (Many [1]) (Many [2]) :: Inl))
