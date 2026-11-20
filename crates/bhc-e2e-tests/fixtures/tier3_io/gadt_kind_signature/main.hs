{-# LANGUAGE GADTs, KindSignatures #-}
module Main where

import Data.Kind (Type)

data Expr :: * -> * where
  IntE  :: Int -> Expr Int
  BoolE :: Bool -> Expr Bool
  Add   :: Expr Int -> Expr Int -> Expr Int
  If    :: Expr Bool -> Expr a -> Expr a -> Expr a
  IsZero :: Expr Int -> Expr Bool

data Tagged :: Type -> Type -> Type where
  Tagged :: String -> a -> Tagged tag a

eval :: Expr a -> a
eval (IntE n) = n
eval (BoolE b) = b
eval (Add a b) = eval a + eval b
eval (If c t e) = if eval c then eval t else eval e
eval (IsZero e) = eval e == 0

label :: Tagged tag a -> String
label (Tagged s _) = s

payload :: Tagged tag a -> a
payload (Tagged _ x) = x

main :: IO ()
main = do
  print (eval (Add (IntE 1) (IntE 2)))
  print (eval (If (IsZero (IntE 0)) (IntE 10) (IntE 20)))
  print (eval (IsZero (Add (IntE 1) (IntE 2))))
  let t = Tagged "answer" (42 :: Int) :: Tagged () Int
  putStrLn (label t)
  print (payload t)
