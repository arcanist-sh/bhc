{-# LANGUAGE TypeOperators #-}
-- Infix data constructor declarations: `data ... = a :* b`. The constructor
-- is the operator; used in construction and pattern matching.
module Main where

data a :* b = a :* b

firstOf :: (a :* b) -> a
firstOf (x :* _) = x

data IntList = Nil | Int :- IntList

total :: IntList -> Int
total Nil       = 0
total (x :- xs) = x + total xs

main :: IO ()
main = do
    print (firstOf (7 :* 8 :: Int :* Int))
    print (total (1 :- (2 :- (3 :- Nil))))
