module Sq (sq, norm, bigger) where

sq :: Num a => a -> a
sq x = x * x

norm :: Num a => a -> a -> a
norm a b = sq a + sq b

bigger :: Ord a => a -> a -> a
bigger a b = if a > b then a else b
