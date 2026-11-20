-- foldM in a monad other than IO. At Maybe or Either the function's result
-- may end the fold; the loop written for IO carried the wrapped value along
-- as the accumulator, and `print (foldM f z xs)` printed an address.
import Control.Monad
import Data.IORef

safeDiv :: Int -> Int -> Maybe Int
safeDiv _ 0 = Nothing
safeDiv a b = Just (a `div` b)

addPositive :: Int -> Int -> Either String Int
addPositive acc x
  | x < 0 = Left ("negative: " ++ show x)
  | otherwise = Right (acc + x)

step :: Int -> Int -> IO Int
step a b = return (a + b)

main :: IO ()
main = do
  print (foldM (\a b -> Just (a + b)) 0 [1, 2, 3 :: Int])
  print (foldM safeDiv 1000 [2, 5, 10])
  print (foldM safeDiv 1000 [2, 0, 10])
  print (foldM safeDiv 7 [])
  print (foldM (\a -> safeDiv a) 100 [5, 2])
  print (foldM addPositive 0 [1, 2, 3])
  print (foldM addPositive 0 [1, -2, 3])
  print (foldM (\acc x -> if x > 2 then Nothing else Just (x : acc)) [] [1, 2 :: Int])
  r <- foldM (\a b -> return (a + b)) 0 [1, 2, 3 :: Int]
  print r
  r2 <- foldM step 10 [1, 2, 3]
  print r2
  ref <- newIORef (0 :: Int)
  r3 <- foldM (\a b -> modifyIORef ref (+ 1) >> return (a * b)) 1 [1, 2, 3, 4]
  n <- readIORef ref
  print (r3, n)
