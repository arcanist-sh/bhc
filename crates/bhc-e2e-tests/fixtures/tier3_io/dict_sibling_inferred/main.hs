module Main where

class Sized a where
  size :: a -> Int

data Box = Box Int

instance Sized Box where
  size (Box n) = n

-- `twice` is polymorphic, so its parameter type for `f` (`a -> Int`) does not
-- pin `a`; the instantiation `a = Box` comes from the sibling value arguments.
-- The dictionary for `sz` is resolved from the whole call's argument types.
twice :: (a -> Int) -> a -> a -> Int
twice f x y = f x + f y

sz :: Sized a => a -> Int
sz x = size x

main :: IO ()
main = print (twice sz (Box 3) (Box 4))
