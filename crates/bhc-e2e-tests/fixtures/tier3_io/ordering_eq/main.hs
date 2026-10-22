-- Ordering values compare by constructor, not by box identity.
data V = A | B | C deriving (Eq, Ord)

main :: IO ()
main = do
  print (LT == LT)
  print (compare A C == LT)
  print (compare C A /= GT)
  print (compare (2 :: Int) 2 == EQ)
  print (GT > LT)
