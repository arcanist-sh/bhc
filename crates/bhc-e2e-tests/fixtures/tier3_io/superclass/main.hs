module Main where

class MyEq a where
    myEqual :: a -> a -> Int

class MyEq a => MyOrd a where
    myCompare :: a -> a -> Int

data Color = Red | Green | Blue

instance MyEq Color where
    myEqual a b = case a of
        Red -> case b of
            Red -> 1
            _ -> 0
        Green -> case b of
            Green -> 1
            _ -> 0
        Blue -> case b of
            Blue -> 1
            _ -> 0

instance MyOrd Color where
    myCompare a b = case a of
        Red -> case b of
            Red -> 0
            _ -> 1
        Green -> case b of
            Red -> 2
            Green -> 0
            Blue -> 1
            _ -> 3
        Blue -> case b of
            Blue -> 0
            _ -> 2

isEqual :: MyOrd a => a -> a -> Int
isEqual x y = myEqual x y

main :: IO ()
main = do
    print (myEqual Red Red)
    print (myEqual Red Green)
    print (myCompare Green Blue)
    print (isEqual Blue Blue)
    print (isEqual Red Green)
