{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

-- stock deriving on a data type
data Color = Red | Green | Blue
    deriving stock (Show, Eq)

-- newtype strategy (GND): delegates to inner type's instances
newtype Wrapper = Wrapper Int
    deriving newtype (Show, Eq)

-- default strategy (no annotation) on a data type
data Shape = Circle | Square | Triangle
    deriving (Show, Eq)

main :: IO ()
main = do
    -- stock Show/Eq on data type
    putStrLn (show Red)
    putStrLn (show Green)
    putStrLn (show (Red == Blue))
    putStrLn (show (Green == Green))
    -- newtype Show/Eq (delegates to Int)
    let w = Wrapper 42
    putStrLn (show w)
    putStrLn (show (w == Wrapper 42))
    putStrLn (show (w == Wrapper 99))
    -- default strategy on data type
    putStrLn (show Circle)
    putStrLn (show (Square == Triangle))
