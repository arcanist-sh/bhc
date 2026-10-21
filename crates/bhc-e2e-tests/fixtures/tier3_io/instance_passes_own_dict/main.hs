{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
-- An instance method that passes its own class dictionary to a helper
-- (pandoc-types' `query f x = f x <> queryInline f x`) merely names itself;
-- its result must reach `<>` evaluated, for Text and list monoids alike.
module Main where

import qualified Data.Text as T
import qualified Data.Text.IO as TIO

class Q a b where
  qry :: Monoid c => (a -> c) -> b -> c

data Leaf = Leaf T.Text | Pair Leaf Leaf

instance Q Leaf Leaf where
  qry f x = f x <> kids f x

kids :: (Q a Leaf, Monoid c) => (a -> c) -> Leaf -> c
kids _ (Leaf _) = mempty
kids f (Pair l r) = qry f l <> qry f r

text :: Leaf -> T.Text
text (Leaf t) = t
text _ = T.empty

count :: Leaf -> [()]
count (Leaf _) = [()]
count _ = []

main :: IO ()
main = do
  TIO.putStrLn (T.pack "[" <> qry text (Leaf (T.pack "img")) <> T.pack "]")
  TIO.putStrLn (T.pack "[" <> qry text (Pair (Leaf (T.pack "a")) (Pair (Leaf (T.pack "b")) (Leaf (T.pack "c")))) <> T.pack "]")
  print (length (qry count (Pair (Leaf T.empty) (Leaf T.empty))))
