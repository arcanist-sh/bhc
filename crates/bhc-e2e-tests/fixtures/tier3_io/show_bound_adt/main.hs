-- `show` on a value held in a VARIABLE, rather than written inline.
-- Each of these printed the value's POINTER as a number, because a Core
-- variable carries no type and `show` fell through to the Int default.
module Main where

data Shape = Circle Int | Square Int deriving Show

mkShape :: IO Shape
mkShape = return (Circle 97)

mkEither :: IO (Either Int Int)
mkEither = return (Right 97)

mkMaybe :: IO (Maybe Int)
mkMaybe = return (Just 97)

viaParam :: Shape -> String
viaParam x = show x

topLevel :: Shape
topLevel = Square 3

main :: IO ()
main = do
  a <- mkShape
  putStrLn (show a)
  b <- mkEither
  putStrLn (show b)
  c <- mkMaybe
  putStrLn (show c)
  let d = Square 5
  putStrLn (show d)
  putStrLn (viaParam (Circle 1))
  putStrLn (show topLevel)
  print a
