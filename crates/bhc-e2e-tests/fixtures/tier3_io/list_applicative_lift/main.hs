-- List Applicative, map over a multi-argument function, liftA2/liftM2 at
-- Maybe/Either/lists/IO, and `for` imported from Data.Traversable (#63).
import Control.Applicative (liftA2)
import Control.Monad (liftM, liftM2)
import Data.Traversable (for)

add :: Int -> Int -> Int
add = (+)

main :: IO ()
main = do
  print ((+) <$> [1, 2] <*> [10, 20 :: Int])
  print ([(+ 1), (* 2)] <*> [5, 6 :: Int])
  print (pure 3 :: [Int])
  print ((,) <$> "ab" <*> [True, False])
  print (map (\f -> f 10) (map add [1, 2]))
  print (map add [1, 2] <*> [10, 20])
  print (liftA2 (+) (Just 1) (Just (2 :: Int)))
  print (liftA2 (+) Nothing (Just (2 :: Int)))
  print (liftA2 (+) [1, 2] [10, 20 :: Int])
  print (liftM2 (+) (Right 1) (Right 2) :: Either String Int)
  print (liftM2 (+) (Left "e") (Right 2) :: Either String Int)
  print (liftM (+ 1) (Just (4 :: Int)))
  a <- liftM2 (+) (putStrLn "first" >> return 1) (putStrLn "second" >> return (2 :: Int))
  print a
  c <- liftA2 (,) (return 'x') (return True)
  print c
  r <- for [1, 2 :: Int] (\x -> return (x * 2))
  print r
