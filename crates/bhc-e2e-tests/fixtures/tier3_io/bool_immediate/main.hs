-- A Bool arrives either boxed or as a tagged immediate. Reading an immediate
-- as a box dereferences address 0 (`show (elem 'b' "abc")` segfaulted), and
-- reading a box as an immediate uses its ADDRESS (`map not` said False twice).
module Main where

main :: IO ()
main = do
  print (elem 'b' "abc")
  print (elem 'z' "abc")
  print (map not [True, False])
  print (filter not [True, False, True])
  print (not (elem 'b' "abc"))
  print (compare (1 :: Int) 2)
