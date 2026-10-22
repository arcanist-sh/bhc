{-# LANGUAGE OverloadedStrings #-}
-- Two dispatch sites inside instance bodies, where typeck records no types:
-- `mappend = (<>)` under a `Semigroup a` constraint must mean `<>` at the
-- instance's own type (not the element's), and a variable bound by a
-- constructor pattern (`Word a`) takes the constructor's field type.
module Main where

import qualified Data.Text as T

data Box a = Box a

unBox :: Box a -> a
unBox (Box x) = x

instance Semigroup a => Semigroup (Box a) where
  Box x <> Box y = Box (x <> y)

instance (Semigroup a, Monoid a) => Monoid (Box a) where
  mempty = Box mempty
  mappend = (<>)

data Tok = Word T.Text | Gap

render :: Tok -> String
render (Word t) = T.unpack t
render Gap = "_"

newtype Toks = Toks [Tok]

instance Semigroup Toks where
  Toks [Word a] <> Toks [Word b] = Toks [Word (a <> b)]
  Toks xs <> Toks ys = Toks (xs ++ ys)

main :: IO ()
main = do
  print (sum (unBox (mappend (Box [1, 2]) (Box [3 :: Int]))))
  putStrLn (unBox (mappend (Box "ab") (Box "cd")))
  let Toks ts = Toks [Word "foo"] <> Toks [Word "bar"]
  print (length ts)
  putStrLn (concatMap render ts)
