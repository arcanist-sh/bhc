-- Boxed Map/Set keys are ordered by content, a stored 0 is found, and
-- `fmap` at Maybe works as a function value (#25).
{-# LANGUAGE OverloadedStrings #-}
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.Text as T

data Color = Red | Green | Blue deriving (Show, Eq, Ord)

main :: IO ()
main = do
  print (Map.toList (Map.fromList [("b", 1 :: Int), ("a", 2), ("c", 3)]))
  print (Map.keys (Map.insert "aa" 0 (Map.fromList [("b", 1 :: Int), ("a", 2)])))
  print (Set.toList (Set.fromList ["pear", "apple", "fig"]))
  print (Map.toList (Map.fromList [(T.pack "zeta", 1 :: Int), (T.pack "alpha", 2)]))
  print (Map.toList (Map.fromList [(3 :: Int, "c"), (1, "a"), (2, "b")]))
  print (Set.toList (Set.fromList [Blue, Red, Green]))
  let m = Map.fromList [(1 :: Int, 0 :: Int), (2, 5)]
  print (Map.lookup 1 m)
  print (Map.findWithDefault 9 1 m)
  print (Map.toList (Map.insertWith (+) 1 3 m))
  print (Map.toList (Map.adjust (+ 1) 1 m))
  print (Map.toList (Map.alter (fmap (+ 1)) 1 m))
  print (Map.toList (Map.alter (const Nothing) 1 m))
  print (Map.toList (Map.unionWith (+) m (Map.fromList [(1, 4)])))
  print (Map.toList (Map.fromListWith (+) [(1 :: Int, 0 :: Int), (1, 0)]))
  let g = fmap (* 2) :: Maybe Int -> Maybe Int
  print (g (Just 4))
  print (map (fmap (+ 1)) [Just 1, Nothing, Just (3 :: Int)])
