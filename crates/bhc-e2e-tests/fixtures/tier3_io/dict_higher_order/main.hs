module Main where

class Sized a where
  size :: a -> Int

data Box = Box Int
data Tag = Tag Int Int

instance Sized Box where
  size (Box n) = n

instance Sized Tag where
  size (Tag a b) = a * b

-- `sz` is a constrained function passed as a VALUE (not applied here) to a
-- higher-order function whose parameter type is concrete. Its dictionary is
-- resolved from that expected parameter type and applied (`sz $dSized_T`).
applyB :: (Box -> Int) -> Box -> Int
applyB f b = f b

applyT :: (Tag -> Int) -> Tag -> Int
applyT f t = f t

sz :: Sized a => a -> Int
sz x = size x

main :: IO ()
main = do
  print (applyB sz (Box 7))
  print (applyT sz (Tag 3 4))
