applyN :: Int -> (Int -> Int) -> Int -> Int
applyN 0 _ x = x
applyN n f x = applyN (n - 1) f (f x)

inc :: Int -> Int
inc x = x + 1

dbl :: Int -> Int
dbl x = x * 2

main :: IO ()
main = do
  print (applyN 3 inc 0)
  print (applyN 4 dbl 1)
