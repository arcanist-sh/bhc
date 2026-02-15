module Main where

class HasLabel a where
    label :: a -> Int
    label _ = 99

data Foo = Foo
  deriving (HasLabel)

main :: IO ()
main = print (label Foo)
