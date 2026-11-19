-- A type with no `Show`, `Eq` or `Ord` instance cannot be shown or
-- compared. `print A` compiled and printed nothing, and `B == B` was True
-- (#97).
data T = A | B

main :: IO ()
main = do
  print A
  print (B == B)
  print (A < B)
