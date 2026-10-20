-- MonadPlus/Alternative at the builtin Maybe and list types.
import Control.Monad (mplus, mzero, msum)
import Control.Applicative ((<|>), empty)
import Data.Foldable (asum)

f :: Int -> Maybe Int
f x = if x > 2 then Just x else Nothing

main :: IO ()
main = do
  print (maybe 0 id (f 1 `mplus` f 5))
  print (maybe 0 id (f 1 <|> f 6))
  print (maybe (-1) id (mzero :: Maybe Int))
  print (maybe 0 id (msum (map f [1, 2, 7, 9])))
  print (maybe 0 id (asum (map f [1, 8])))
  print (maybe (-1) id (msum (map f [1, 2])))
  print (sum ([1, 2] `mplus` [3 :: Int]))
  print (length ([1 :: Int] <|> [4, 5]))
  print (sum (msum [[1, 2], [10 :: Int]]))
  print (length (mzero :: [Int]))
