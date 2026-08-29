-- `not` on a boxed Bool. The old lowering XORed a bit of the ADDRESS, so
-- `not False` came back as a garbage pointer that tested as false and
-- `if not False` took the wrong branch.
module Main where

main :: IO ()
main = do
  putStrLn (if not False then "not False is True" else "WRONG")
  putStrLn (if not True then "WRONG" else "not True is False")
  putStrLn (if not (not True) then "double negation holds" else "WRONG")
  print (not True)
  print (not False)
  print (not (1 > 2))
