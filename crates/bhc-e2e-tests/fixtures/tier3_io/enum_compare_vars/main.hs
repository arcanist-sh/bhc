-- Derived Eq/Ord on an enumeration compares constructors, also when both
-- operands are variables (no constructor in sight to name the type).
data Level = Low | Mid | High deriving (Eq, Ord, Show)

level :: Int -> Level
level 0 = Low
level 1 = Mid
level _ = High

main :: IO ()
main = do
  let x = level 1
      y = level 1
      z = level 2
  print (x == y)
  print (x /= y)
  print (x <= y)
  print (z <= x)
  print (x < z)
  print (z > y)
  putStrLn (if x == y then "same" else "different")
