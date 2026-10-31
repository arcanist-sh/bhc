-- Derived Show parenthesizes constructor applications and negative fields.
data Tree = Leaf | Node Tree Int Tree deriving Show
data P = P Int Int deriving Show
data W = W (Maybe Int) [Int] String Char (Int, Bool) deriving Show
main :: IO ()
main = do
  print (Node Leaf 3 (Node Leaf 4 Leaf))
  print (P (-1) 2)
  print (W (Just 1) [1] "a b" 'x' (1, True))
  print (Just (Node Leaf 1 Leaf))
