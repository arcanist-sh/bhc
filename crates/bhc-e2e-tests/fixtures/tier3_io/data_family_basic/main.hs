{-# LANGUAGE TypeFamilies #-}
module Main where

data family Container a

data instance Container Int = IntContainer Int Int
data instance Container Bool = BoolContainer Bool

getIntFirst :: Container Int -> Int
getIntFirst (IntContainer x _) = x

getBool :: Container Bool -> Bool
getBool (BoolContainer b) = b

main :: IO ()
main = do
  let ic = IntContainer 42 99
  let bc = BoolContainer True
  putStrLn (show (getIntFirst ic))
  putStrLn (show (getBool bc))
