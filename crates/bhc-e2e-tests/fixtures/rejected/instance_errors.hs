-- Instances must match their class. Each of these compiled (#97): a method
-- whose body has the wrong type, a method the class does not declare, and
-- class methods used at types the class has no instance for.
class Size a where
  size :: a -> Int

instance Size Int where
  size n = n > 0

instance Size Bool where
  size _ = 1
  weight _ = 2

class Box f where
  unbox :: f a -> a

instance Box Maybe where
  unbox (Just x) = x
  unbox Nothing = error "empty"

main :: IO ()
main = do
  print (size 'c')
  print (unbox [3 :: Int])
