-- `Control.Arrow`'s `(***)` and `(&&&)` work at the function arrow.
-- pandoc-types' `definitionList` maps `toList *** map toList` over its
-- items; with no implementation both combinators ran as stubs.
module Main where
import Control.Arrow ((***), (&&&))

pairs :: [(Int, [Int])]
pairs = map ((+ 1) *** map (* 2)) [(1, [1, 2]), (3, [4])]

main :: IO ()
main = do
  print (map fst pairs)
  print (map (sum . snd) pairs)
  let (a, b) = (length &&& sum) [1, 2, 3 :: Int]
  print a
  print b
  let (c, d) = ((* 10) *** show) (4 :: Int, True)
  print c
  putStrLn d
