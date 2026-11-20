{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- A newtype's constructor is erased, so `N 1` is the bare Int. Comparing
-- two of them passed the unboxed field to the derived comparison, which
-- takes a pointer: the module did not compile.
newtype N = N Int deriving (Show, Eq, Ord)
newtype D = D Double deriving (Show, Eq, Ord)

main :: IO ()
main = do
  print (N 1 < N 2, N 2 <= N 1, N 3 > N 3, N 3 >= N 3)
  print (N 1 == N 1, N 1 /= N 1, N 1 == N 2)
  print (compare (N 1) (N 2), compare (N 2) (N 2))
  print (max (N 1) (N 2), min (N 1) (N 2))
  print (D 1.5 < D 2.5, D 1.5 == D 1.5, max (D 1.5) (D 0.5))
