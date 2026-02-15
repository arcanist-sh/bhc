module Main where

class HasInfo a where
    name :: a -> String
    name _ = "unnamed"
    priority :: a -> Int
    priority _ = 0

data Widget = Widget
  deriving (HasInfo)

main :: IO ()
main = do
    putStrLn (name Widget)
    print (priority Widget)
