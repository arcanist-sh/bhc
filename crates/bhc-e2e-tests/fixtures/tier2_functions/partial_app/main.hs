add :: Int -> Int -> Int
add x y = x + y

add3 :: Int -> Int -> Int -> Int
add3 x y z = x + y + z

add5 :: Int -> Int
add5 = add 5

applyN :: Int -> (Int -> Int) -> Int -> Int
applyN 0 _ x = x
applyN n f x = applyN (n - 1) f (f x)

main :: IO ()
main = do
  print (applyN 3 (add 10) 0)
  print (applyN 2 (add3 1 2) 0)
  print (add5 3)
