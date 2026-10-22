{-# LANGUAGE TupleSections #-}
-- `x :| xs` builds a value that list code reads (bhc models `NonEmpty a` as
-- `[a]`): built, matched, and mapped with `<$>`. pandoc's AnnotatedTable
-- builds a cell's column specs as `snd <$> chStart :| chRest` and the HTML
-- writer matches `(colspec :| _)`; `:|` used to compile to a stub.
module Main where
import Data.List.NonEmpty (NonEmpty (..))

neHead :: NonEmpty a -> a
neHead (x :| _) = x

neTail :: NonEmpty a -> [a]
neTail (_ :| xs) = xs

pairs :: NonEmpty (Int, String)
pairs = (1, "a") :| [(2, "b"), (3, "c")]

main :: IO ()
main = do
  let ne = 1 :| [2, 3 :: Int]
  print (neHead ne)
  print (neTail ne)
  print (neHead (fmap (* 10) ne))
  print (neTail ((+ 1) <$> ne))
  putStrLn (neHead (snd <$> pairs))
  print (map fst (neTail pairs))
  print (length (neTail ((, 'x') <$> ne)))
