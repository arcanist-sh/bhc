-- An associated type at a list instance used at several element types, and
-- instances declared at `String` found at `[Char]` (#108).
{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FunctionalDependencies, FlexibleInstances #-}
module Main where

class Container f where
  type Elem f
  empty  :: f
  insert :: Elem f -> f -> f
  toL    :: f -> [Elem f]

instance Container [a] where
  type Elem [a] = a
  empty = []
  insert = (:)
  toL = id

class Convert a b | a -> b where
  convert :: a -> b

instance Convert Int String where
  convert = show

instance Convert Bool Int where
  convert b = if b then 1 else 0

instance Convert Char [Char] where
  convert c = [c, c]

class Describe a where
  describe :: a -> String

instance Describe String where
  describe s = "string " ++ s

instance Describe [Int] where
  describe xs = "ints " ++ show (sum xs)

instance Describe a => Describe (Maybe a) where
  describe Nothing = "nothing"
  describe (Just x) = "just " ++ describe x

main :: IO ()
main = do
  print (toL (insert (1 :: Int) (insert 2 empty) :: [Int]))
  print (toL (insert 2 (empty :: [Int])))
  print (toL (insert 'x' (insert 'y' empty) :: String))
  putStrLn (convert (42 :: Int))
  putStrLn (convert (7 :: Int) ++ "!")
  print (convert True + 1)
  putStrLn (convert 'q')
  putStrLn (describe "abc")
  putStrLn (describe ("a" ++ "b"))
  putStrLn (describe [1, 2, 3 :: Int])
  putStrLn (describe (Just "x"))
  putStrLn (describe (Just [4, 5 :: Int]))
  print (toL (insert (1 :: Int) (empty :: [Int])))
  print (toL (insert 'x' (insert 'y' empty) :: [Char]))
  print (toL (insert True (empty :: [Bool])))
