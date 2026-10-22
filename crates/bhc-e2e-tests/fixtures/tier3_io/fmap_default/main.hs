-- fmapDefault is fmap for a lawful Traversable (pandoc-types Walkable uses it).
import qualified Data.Traversable as T
import Data.Traversable (fmapDefault)
main :: IO ()
main = do
  print (sum (T.fmapDefault (* 2) [1, 2, 3 :: Int]))
  print (sum (fmapDefault (+ 1) [1, 2, 3 :: Int]))
