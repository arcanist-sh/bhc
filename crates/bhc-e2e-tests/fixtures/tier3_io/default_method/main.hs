module Main where

class HasDefault a where
    value :: a -> Int
    doubled :: a -> Int
    doubled x = value x * 2

data MyData = MyData Int

instance HasDefault MyData where
    value (MyData n) = n

main :: IO ()
main = do
    print (value (MyData 21))
    print (doubled (MyData 21))
