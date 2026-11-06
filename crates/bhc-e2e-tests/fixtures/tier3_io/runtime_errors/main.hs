-- Division rounds as GHC does, and runtime errors in pure code are exceptions that
-- `try` sees (#76).
import Control.Exception
import Data.Maybe (fromJust)
import Data.Ratio

half :: Int -> Int -> Int
half x y = x `div` y

main :: IO ()
main = do
  print ((-7) `div` 2 :: Int, (-7) `mod` 2 :: Int, (-7) `quot` 2 :: Int, (-7) `rem` 2 :: Int)
  print (7 `div` (-2) :: Int, 7 `mod` (-2) :: Int, 7 `quot` (-2) :: Int, 7 `rem` (-2) :: Int)
  print (divMod (-7) 2 :: (Int, Int), quotRem (-7) 2 :: (Int, Int))
  let big = -7 :: Integer
  print (big `div` 2, big `mod` 2, big `quot` 2, big `rem` 2)
  a <- try (evaluate (1 `div` (0 :: Int))) :: IO (Either ArithException Int)
  putStrLn (either show show a)
  b <- try (evaluate (half 10 0)) :: IO (Either ArithException Int)
  putStrLn (either show show b)
  c <- try (evaluate (10 `mod` (0 :: Int))) :: IO (Either ArithException Int)
  putStrLn (either show show c)
  d <- try (evaluate (10 `quot` (0 :: Int))) :: IO (Either ArithException Int)
  putStrLn (either show show d)
  e <- try (evaluate (10 `rem` (0 :: Int))) :: IO (Either ArithException Int)
  putStrLn (either show show e)
  f <- try (evaluate (fst (divMod 10 (0 :: Int)))) :: IO (Either ArithException Int)
  putStrLn (either show show f)
  g <- try (evaluate (10 `div` (0 :: Integer))) :: IO (Either ArithException Integer)
  putStrLn (either show show g)
  h <- try (evaluate (head ([] :: [Int]))) :: IO (Either SomeException Int)
  putStrLn (either (const "head failed") show h)
  i <- try (evaluate (tail ([] :: [Int]))) :: IO (Either SomeException [Int])
  putStrLn (either (const "tail failed") show i)
  j <- try (evaluate (last ([] :: [Int]))) :: IO (Either SomeException Int)
  putStrLn (either (const "last failed") show j)
  k <- try (evaluate (init ([] :: [Int]))) :: IO (Either SomeException [Int])
  putStrLn (either (const "init failed") show k)
  l <- try (evaluate ([1, 2 :: Int] !! 5)) :: IO (Either SomeException Int)
  putStrLn (either (const "index failed") show l)
  m <- try (evaluate (fromJust (Nothing :: Maybe Int))) :: IO (Either SomeException Int)
  putStrLn (either (const "fromJust failed") show m)
  n <- try (evaluate (undefined :: Int)) :: IO (Either SomeException Int)
  putStrLn (either (const "undefined failed") show n)
  o <- try (evaluate (1 % (0 :: Integer))) :: IO (Either ArithException Rational)
  putStrLn (either show show o)
  print (head [1, 2 :: Int], 10 `div` (2 :: Int))
  e2 <- try (throwIO (ErrorCall "ec")) :: IO (Either ErrorCall ())
  putStrLn (either show (const "right") e2)
  print (1 % (3 :: Integer))
