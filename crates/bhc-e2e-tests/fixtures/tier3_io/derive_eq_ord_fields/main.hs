data Shape = Circle Int | Square Int deriving (Eq, Ord, Show)
data Pair = Pair Int Int deriving (Eq, Ord, Show)

main :: IO ()
main = do
  -- Ord/compare on constructors with fields
  print (compare (Circle 5) (Circle 3))   -- GT (field)
  print (compare (Circle 3) (Circle 5))   -- LT (field)
  print (compare (Circle 9) (Square 1))   -- LT (tag 0 < 1)
  print (Circle 5 < Square 0)             -- True
  print (Circle 5 < Circle 3)             -- False
  -- Eq on constructors with fields
  print (Circle 5 == Circle 5)            -- True
  print (Circle 5 == Circle 3)            -- False
  print (Circle 5 /= Circle 3)            -- True
  print (Square 2 == Circle 2)            -- False
  -- multi-field lexicographic
  print (compare (Pair 1 2) (Pair 1 3))   -- LT
  print (compare (Pair 2 1) (Pair 1 9))   -- GT
  print (Pair 1 2 == Pair 1 2)            -- True
  print (Pair 3 4 <= Pair 3 4)            -- True
