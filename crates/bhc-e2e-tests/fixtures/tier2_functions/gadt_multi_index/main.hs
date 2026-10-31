{-# LANGUAGE GADTs #-}
-- A typed expression evaluator: a GADT indexed at several types (#33).
data Expr a where
  IntE  :: Int -> Expr Int
  BoolE :: Bool -> Expr Bool
  Add   :: Expr Int -> Expr Int -> Expr Int
  Eq    :: Expr Int -> Expr Int -> Expr Bool
  If    :: Expr Bool -> Expr a -> Expr a -> Expr a
  Pair  :: Expr a -> Expr b -> Expr (a, b)
  Fst   :: Expr (a, b) -> Expr a

eval :: Expr a -> a
eval (IntE n) = n
eval (BoolE b) = b
eval (Add x y) = eval x + eval y
eval (Eq x y) = eval x == eval y
eval (If c t e) = if eval c then eval t else eval e
eval (Pair x y) = (eval x, eval y)
eval (Fst p) = fst (eval p)

size :: Expr a -> Int
size e = case e of
  IntE _ -> 1
  BoolE _ -> 1
  Add x y -> 1 + size x + size y
  Eq x y -> 1 + size x + size y
  If c t f -> 1 + size c + size t + size f
  Pair x y -> 1 + size x + size y
  Fst p -> 1 + size p

describe :: Int -> Expr a -> String
describe n (IntE k) = "int " ++ show (n + k)
describe _ (BoolE b) = "bool " ++ show b
describe _ _ = "other"

main :: IO ()
main = do
  print (eval (If (BoolE True) (IntE 1) (IntE 2)))
  print (eval (Add (IntE 3) (IntE 4)))
  print (eval (Eq (Add (IntE 1) (IntE 1)) (IntE 2)))
  print (eval (If (Eq (IntE 1) (IntE 2)) (BoolE False) (BoolE True)))
  print (eval (Pair (IntE 5) (BoolE False)))
  print (eval (Fst (Pair (Add (IntE 2) (IntE 3)) (BoolE True))))
  print (size (If (BoolE True) (Add (IntE 1) (IntE 2)) (IntE 3)))
  putStrLn (describe 10 (IntE 5))
  putStrLn (describe 0 (BoolE True))
