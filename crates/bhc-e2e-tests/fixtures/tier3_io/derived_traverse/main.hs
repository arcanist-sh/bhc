-- traverse over derived Traversable types at Maybe and Either (#32).
{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
data Pair a = Pair a a deriving (Show, Functor, Foldable, Traversable)
data Box a = Box a deriving (Show, Functor, Foldable, Traversable)
data Tagged a = Tagged Int a | Untagged deriving (Show, Functor, Foldable, Traversable)

half :: Int -> Maybe Int
half n = if even n then Just (n `div` 2) else Nothing

check :: Int -> Either String Int
check n = if n > 0 then Right n else Left ("bad " ++ show n)

main :: IO ()
main = do
  print (traverse Just (Pair 1 (2 :: Int)))
  print (traverse half (Pair 4 6))
  print (traverse half (Pair 4 5))
  print (traverse half (Box 8))
  print (traverse half (Tagged 7 10))
  print (traverse half (Untagged :: Tagged Int))
  print (traverse check (Pair 1 2))
  print (traverse check (Pair 1 (-2)))
