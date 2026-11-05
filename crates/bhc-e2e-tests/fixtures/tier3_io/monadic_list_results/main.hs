-- The results of `zipWithM` and `filterM` print as lists (#73), and `()` prints.
import Control.Monad

main :: IO ()
main = do
  z <- zipWithM (\a b -> return (a + b)) [1, 2] [3, 4 :: Int]
  print z
  f <- filterM (\x -> return (x > 1)) [1, 2, 3 :: Int]
  print f
  h <- filterM (\x -> return (x > (1 :: Int))) [1, 2, 3]
  print h
  j <- filterM (const (return True)) "ab"
  print j
  print ()
  return () >>= print
