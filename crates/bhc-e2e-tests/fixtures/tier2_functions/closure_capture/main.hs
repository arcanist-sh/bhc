applyN :: Int -> (Int -> Int) -> Int -> Int
applyN 0 _ x = x
applyN n f x = applyN (n - 1) f (f x)

main :: IO ()
main = do
  let k = 100
  let f = \x -> x + k
  print (applyN 3 f 0)
