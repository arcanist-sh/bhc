{-# LANGUAGE DeriveFunctor #-}
-- `print` of a derived-Show value produced by a call (#32).
data Pair a = Pair a a deriving (Show, Functor)

data Box a = Box a deriving (Show, Functor)

data Tree a = Leaf | Node (Tree a) a (Tree a) deriving (Show, Functor)

data Rose a = Rose a [Rose a] deriving (Show, Functor)

data Opt a = Opt (Maybe a) [a] deriving (Show, Functor)

main :: IO ()
main = do
  print (fmap (+ 1) (Pair 1 (2 :: Int)))
  print (fmap (+ 1) (Box (1 :: Int)))
  case fmap (+ 1) (Pair 1 (2 :: Int)) of
    Pair a b -> print (a + b)
  print (fmap (* 2) (Node Leaf (3 :: Int) (Node Leaf 4 Leaf)))
  print (fmap (+ 1) (Rose (1 :: Int) [Rose 2 [], Rose 3 []]))
  print (fmap (+ 1) (Opt (Just (1 :: Int)) [2, 3]))
