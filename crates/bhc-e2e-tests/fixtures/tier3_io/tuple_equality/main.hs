-- Tuples compare component-wise by type, not by box address.
main :: IO ()
main = do
  let r = foldr (\x (lo, hi) -> (min x lo, max x hi)) (100, 0) [7, 3, 9 :: Int]
  print ((3, 9) == (3 :: Int, 9 :: Int))
  print (r == (3, 9))
  print (r /= (3, 9))
  print ((1 :: Int, "ab") == (1, "ab"))
  print ((1 :: Int, "ab") == (1, "ac"))
  print ((True, 'x') == (True, 'x'))
  print (((1 :: Int, 2 :: Int), 3 :: Int) == ((1, 2), 3))
  print (((1 :: Int, 2 :: Int), 3 :: Int) == ((1, 5), 3))
