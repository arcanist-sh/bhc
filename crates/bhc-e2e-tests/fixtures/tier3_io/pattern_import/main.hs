{-# LANGUAGE PatternSynonyms #-}
module Main (pattern Zero, pattern One, main) where

data Expr = Lit Int | Add Expr Expr

pattern Zero = Lit 0
pattern One  = Lit 1

eval :: Expr -> Int
eval e = case e of
  Lit n   -> n
  Add a b -> eval a + eval b

isZero :: Expr -> String
isZero e = case e of
  Zero -> "zero"
  _    -> "not zero"

main :: IO ()
main = do
  putStrLn (isZero Zero)
  putStrLn (isZero One)
  putStrLn (show (eval (Add One One)))
