-- zip3 / zipWith3 carry their element types (show, destructuring).
main :: IO ()
main = do
  print (zip3 [1, 2, 3 :: Int] "ab" [True, False, True])
  let z = zip3 [1 :: Int ..] "ab" [True, False]
  print (length z)
  mapM_ (\(a, b, c) -> print a >> print b >> print c) z
  print (zipWith3 (\a b c -> a + b + c) [1, 2 :: Int] [10, 20] [100, 200])
  print (zipWith3 (\a b c -> (a, b, c)) "xy" [Just 1, Nothing :: Maybe Int] [2.5, 3.5 :: Double])
