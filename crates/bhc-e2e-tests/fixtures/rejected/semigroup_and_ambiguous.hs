-- Two programs that type-checked and then failed in code generation:
-- `<>` and `mconcat` at a type with no Semigroup or Monoid instance, and a
-- number used as a container, whose type nothing determines (#97).
main :: IO ()
main = do
  print (True <> False)
  print ((1 :: Int) <> 2)
  print (mconcat "ab")

container :: Int
container = length 1
