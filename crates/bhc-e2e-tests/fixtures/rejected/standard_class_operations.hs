-- Standard class operations at a type with no instance of the class. Each
-- compiled: the builtin types of `succ`, `even`, `/` and `div` carry no
-- class, so `succ "x"` and `(2 :: Int) / 3` ran on the wrong representation
-- (#97).
main :: IO ()
main = do
  print (succ "x")
  print (even 'c')
  print ((2 :: Int) / 3)
  print (div (2.5 :: Double) 2)
  print (round (4 :: Int) :: Integer)
