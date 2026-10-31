-- An annotation fixes the type of a whole operator chain, and a list
-- comprehension `let` takes its generator's type: defaulting must not
-- overwrite a type that is already known (#57). An integer literal pattern
-- at `Double` is the float.
classify :: Double -> String
classify 0 = "zero"
classify 1 = "one"
classify x = show x

main :: IO ()
main = do
  print (2 ^ 64 + 5 :: Integer)
  print ((2 ^ 64) + 5 :: Integer)
  print (10000000000 * 10000000000 + 1 :: Integer)
  print (2 ^ 10 + 1 :: Int)
  print (3 + 4)
  print [y | x <- [1, 2 :: Integer], let y = x + 10]
  print [y | x <- [1, 2 :: Integer], let y = x * 10000000000 * 10000000000]
  mapM_ (putStrLn . classify) [0, 1, 2.5]
