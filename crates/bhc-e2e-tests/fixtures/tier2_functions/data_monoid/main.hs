-- Data.Monoid newtypes and their instances (#52).
import Data.Monoid

main :: IO ()
main = do
  print (getSum (Sum 3 <> Sum (4 :: Int)))
  print (getProduct (mconcat (map Product [1 .. 5 :: Int])))
  print (getAll (foldMap (All . even) [2, 4 :: Int]), getAny (foldMap (Any . odd) [2, 4 :: Int]))
  print (getFirst (First (Just 'a') <> First Nothing <> First (Just 'c')))
  print (getLast (mconcat [Last (Just 1), Last Nothing, Last (Just (3 :: Int))]))
  print (Sum (2 :: Int), mempty :: Product Int)
  print (All True <> mempty, Any False)
  print (getSum (mempty :: Sum Int))
