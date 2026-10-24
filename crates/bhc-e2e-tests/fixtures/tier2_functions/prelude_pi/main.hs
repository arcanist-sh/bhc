-- `pi` is in scope from the Prelude (issue #19).
main :: IO ()
main = do
  print (pi :: Double)
  print (2 * pi * 1.5 :: Double)
