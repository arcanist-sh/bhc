-- Derived Show of fields whose type is (or mentions) a type parameter (#59).
import Shapes

data D a = D a deriving Show
data P a b = P a b deriving Show
newtype S a = S { getS :: a } deriving Show
data Tree a = Leaf | Node (Tree a) a (Tree a) deriving Show
data W a = W [a] (Maybe a) deriving Show

main :: IO ()
main = do
  print (D True)
  print (D 'x', D "str", D (2.5 :: Double))
  print (P 'a' False)
  print (S 'x')
  print (Node Leaf 'q' (Node Leaf 'r' Leaf))
  print (W "ab" (Just 'c'))
  print [D True, D False]
  print (Just (D (Just 'z')))
  putStrLn (show (D True))
  print (D (3 :: Int), D (-4 :: Int))
  print (Box True, Box "boxed")
