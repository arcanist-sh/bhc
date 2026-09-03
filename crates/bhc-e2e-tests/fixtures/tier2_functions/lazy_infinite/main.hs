-- A finite consumer of an infinite producer must terminate.
main :: IO ()
main = do
  print (take 3 (repeat (7 :: Int)))
  print (take 4 (cycle [1, 2 :: Int]))
  print (head (filter (> 10) [1 :: Int ..]))
  putStrLn "infinite ok"
