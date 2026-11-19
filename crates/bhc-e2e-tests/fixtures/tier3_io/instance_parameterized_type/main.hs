-- A hand-written Show, Eq or Ord instance for a type with a parameter.
-- Its symbol names the head's type variable (`$instance_show_C_v`) and was
-- never found for `C`: `show` printed an address, `print` an empty line,
-- and `==` was False.
data C a = C a

instance Show a => Show (C a) where
  show (C x) = "C " ++ show x

instance Eq a => Eq (C a) where
  C x == C y = x == y

instance Ord a => Ord (C a) where
  compare (C x) (C y) = compare x y

data Pair a b = Pair a b

instance (Show a, Show b) => Show (Pair a b) where
  show (Pair x y) = "<" ++ show x ++ "|" ++ show y ++ ">"

main :: IO ()
main = do
  putStrLn (show (C (3 :: Int)))
  print (C (4 :: Int))
  print (C (5 :: Int), True)
  print (C 'x' == C 'x', C 'x' == C 'y')
  print (compare (C (1 :: Int)) (C 2), C (2 :: Int) < C 1)
  print (Pair (1 :: Int) (2 :: Int))
