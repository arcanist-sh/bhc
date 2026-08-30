-- `Maybe`, `Either` and `[]` are monads but not transformer layers, so a
-- do-block in one was lowered as if it were IO: `x <- half n` bound `x` to the
-- whole `Just`, nothing short-circuited, and `return` was the identity.
module Main where

half :: Int -> Maybe Int
half n = if even n then Just (n `div` 2) else Nothing

chainM :: Int -> Maybe Int
chainM n = do
  x <- half n
  y <- half x
  return (x + y)

safeDiv :: Int -> Int -> Either String Int
safeDiv _ 0 = Left "div0"
safeDiv a b = Right (a `div` b)

chainE :: Int -> Either String Int
chainE n = do
  x <- safeDiv 100 n
  y <- safeDiv x 2
  return (x + y)

pairs :: [Int]
pairs = do
  x <- [1, 2]
  y <- [10, 20]
  return (x + y)

describe :: Maybe Int -> String
describe Nothing  = "nothing"
describe (Just n) = "just " ++ show n

describeE :: Either String Int -> String
describeE (Left e)  = "left " ++ e
describeE (Right n) = "right " ++ show n

main :: IO ()
main = do
  putStrLn (describe (chainM 8))
  putStrLn (describe (chainM 6))
  putStrLn (describe (chainM 3))
  putStrLn (describeE (chainE 5))
  putStrLn (describeE (chainE 0))
  print pairs
  putStrLn (describe (Just (1 :: Int) >>= \x -> Just (x + 1)))
  putStrLn (describe (Nothing >>= \x -> Just (x + (1 :: Int))))
