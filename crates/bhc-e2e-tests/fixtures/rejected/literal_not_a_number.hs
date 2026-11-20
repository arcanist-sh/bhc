-- A numeric literal used as a list, a tuple, a function or an IO action:
-- each asks for a `Num` instance those types do not have. They compiled,
-- because the constraint still mentioned a type variable (#97).
main :: IO ()
main = do
  print (head 1)
  print (fst 1)
  print (map 1 [True])
  x <- 3
  print (x :: Int)
