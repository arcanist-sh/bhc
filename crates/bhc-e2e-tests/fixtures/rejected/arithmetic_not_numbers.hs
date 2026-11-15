-- Arithmetic at Char, Bool and a tuple has no Num instance (#97).
main :: IO ()
main = do
  print (negate (succ 'a'))
  print (abs True)
  print ((1, 2) + (3, 4))
  print (fromIntegral 'a' :: Int)
