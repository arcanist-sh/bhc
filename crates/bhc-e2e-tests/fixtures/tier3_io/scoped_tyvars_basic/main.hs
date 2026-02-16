{-# LANGUAGE ScopedTypeVariables #-}
module Main where

f :: forall a. a -> (a, a)
f x = let y = x :: a in (y, y)

main :: IO ()
main = do
    let pair = f (42 :: Int)
    print (fst pair)
    print (snd pair)
