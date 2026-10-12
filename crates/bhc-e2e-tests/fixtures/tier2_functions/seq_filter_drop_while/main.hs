module Main where
import qualified Data.Sequence as Seq
import Data.Foldable (toList)

isSmall :: Int -> Bool
isSmall x = x < 3

main :: IO ()
main = do
  let s = Seq.fromList [1, 2, 5, 1, 2 :: Int]
  print (toList (Seq.filter isSmall s))
  print (toList (Seq.dropWhileL isSmall s))
  print (toList (Seq.dropWhileR isSmall s))
  print (toList (Seq.dropWhileL (< 9) s))
  print (toList (Seq.dropWhileL isSmall (Seq.dropWhileR isSmall s)))
