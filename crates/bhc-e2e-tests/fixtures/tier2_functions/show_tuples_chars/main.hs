-- Show of 3+-tuples, tuples built in lambdas, Char results, negative
-- numbers under constructors (#24).
import Data.Char (chr)

class Cmp a where
  cmp :: a -> a -> Ordering

instance Cmp Int where
  cmp x y = compare x y

main :: IO ()
main = do
  print (1 :: Int, True, 'c')
  print (2 :: Int, "two", [3, 4 :: Int], Just False)
  putStrLn (show (1.5 :: Double, 'x', ()))
  print [(1 :: Int, 'a', True), (2, 'b', False)]
  print (map (\x -> (x, True)) [1, 2 :: Int])
  print (chr 66)
  print (maximum "hello")
  print (Just (-3 :: Int))
  print [Left (-1.5), Right 2 :: Either Double Int]
  mapM_ print "hi"
  mapM_ print [(1 :: Int, True)]
  print (cmp (1 :: Int) 2)
  print [cmp (2 :: Int) 1, cmp 3 3]
