import qualified Data.Map as Map
import qualified Data.Set as Set

data Color = Red | Green | Blue deriving (Show, Eq, Ord)

main :: IO ()
main = do
  let m = Map.fromList [(3, "c"), (1, "a"), (2, "b")] :: Map.Map Int String
  print m
  putStrLn (show m)
  print (Map.empty :: Map.Map Int Int)
  let s = Set.fromList "hello"
  print s
  print (Set.fromList [3, 1, 2 :: Int])
  print (Map.fromList [("x", Just True), ("a", Nothing)])
  print (Just (Set.fromList [Red, Blue]))
  print (Map.insert 5 [1,2] (Map.singleton 4 [3 :: Int]))
  putStrLn ("m = " ++ show (Map.map length m))
