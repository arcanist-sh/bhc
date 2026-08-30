-- `Data.Foldable.toList` had no implementation at all: pandoc's `B.doc` is
-- `Pandoc nullMeta . toList`, so `readMarkdown` aborted with
-- "stub: Data.Foldable.toList not implemented".
module Main where
import Data.Foldable (toList)
import qualified Data.Sequence as Seq

main :: IO ()
main = do
  print (toList (Seq.fromList [1, 2, 3 :: Int]))
  print (toList (Seq.empty :: Seq.Seq Int))
  print (toList [4, 5 :: Int])
  print (length (toList (Seq.fromList "abc")))
