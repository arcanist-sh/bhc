module Main where

class Printable a where
    display :: a -> String
    label   :: a -> String

data Shape = Circle | Square

instance Printable Shape where
    display Circle = "O"
    display Square = "#"
    label Circle   = "circle"
    label Square   = "square"

showBoth :: Printable a => a -> String
showBoth x = label x ++ ": " ++ display x

main :: IO ()
main = do
    putStrLn (showBoth Circle)
    putStrLn (showBoth Square)
