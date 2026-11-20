-- max and min order strings, lists, tuples, Doubles and Integers by their
-- contents, as the comparison operators do; they compared the two words.
-- compare on two variables of a type with a derived Ord uses that ordering:
-- with no constructor in sight it compared the two objects by address.
{-# LANGUAGE DeriveFunctor #-}
data Tree a = Leaf | Node (Tree a) a (Tree a)
  deriving (Show, Eq, Ord, Functor)

t :: Tree Int
t = Node (Node Leaf 1 Leaf) 2 (Node Leaf 3 Leaf)

main :: IO ()
main = do
  print (max "apple" "banana", min "apple" "banana")
  print (max 2.5 (-1.5) :: Double, min 2.5 (-1.5) :: Double)
  print (max (1, 'b') (1 :: Int, 'c'), min [3, 1] [2, 9 :: Int])
  print (max 3 (7 :: Int), min 'a' 'b', max (10 :: Integer) (2 ^ 70))
  print (max 1.5 (2 :: Float), max (Just 1) (Nothing :: Maybe Int))
  print (compare t (fmap (+ 1) t))
  print (compare (Node Leaf (1 :: Int) Leaf) (fmap (+ 1) (Node Leaf 1 Leaf)))
  print (compare (Node Leaf (1 :: Int) Leaf) (fmap (+ 1) (Node Leaf 0 Leaf)))
  print (fmap (+ 1) t == t, fmap (+ 0) t == t, fmap (+ 1) t)
  let u = fmap (+ 1) t
  print (compare t u, compare u t, t < u)
  print (widths [0.5, 4, 0.25], map (max 1 . fromIntegral) [0, 3 :: Int] :: [Double])
 where
  -- One operand a variable, the other a constant.
  widths :: [Double] -> [Double]
  widths ws = map (\w -> max w 1.0) ws ++ [min 2 (sum ws)]
