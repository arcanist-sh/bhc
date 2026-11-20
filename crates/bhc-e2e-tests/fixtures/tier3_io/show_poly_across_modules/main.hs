-- A `Show`-constrained function from another module, used at types other than
-- the numeric ones. Only numeric uses got a copy of the library body, so its
-- `show` ran with no type and printed the value's machine word.
import qualified Data.Text.IO as TIO
import ShowLib

data Colour = Red | Green deriving Show

main :: IO ()
main = do
  putStrLn (label Nothing True)
  putStrLn (label (Just 1) (3 :: Int))
  putStrLn (label Nothing (Just False))
  putStrLn (label Nothing LT)
  putStrLn (label Nothing Green)
  putStrLn (label Nothing "text")
  TIO.putStrLn (labelled (Just Red))
  putStrLn (bracketed [True, False])
  putStrLn (bracketed [2.5 :: Double])
