module Main where

class Sized a where
  size :: a -> Int

data Box = Box Int
data Pair = Pair Int Int

instance Sized Box where
  size (Box n) = n

instance Sized Pair where
  size (Pair a b) = a + b

-- Recursive, polymorphic over `Sized a`: the dictionary cannot be erased by
-- inlining/specialization, so it must be passed as a runtime argument and
-- threaded through the recursive call. Each `main` call site resolves a
-- different instance dictionary (Box vs Pair).
sumSizes :: Sized a => [a] -> Int
sumSizes [] = 0
sumSizes (x:xs) = size x + sumSizes xs

main :: IO ()
main = do
  print (sumSizes [Box 3, Box 4, Box 5])
  print (sumSizes [Pair 1 2, Pair 3 4])
