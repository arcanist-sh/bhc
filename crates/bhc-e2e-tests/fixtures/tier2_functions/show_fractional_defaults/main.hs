-- `show` of a fractional value whose type defaults: Haskell defaults to
-- Double. bhc defaulted to Float, whose show path read the f64 value as f32
-- bits and printed "0" (or garbage, for `show (sqrt 2)` and `show pi`).
main :: IO ()
main = do
  putStrLn (show 3.0)
  print (show 2.5, show 3.0)
  putStrLn (show (sqrt 2))
  putStrLn (show pi)
  putStrLn (show (1.5 + 2))
  let x = 2.5
  putStrLn (show x)
  putStrLn (show (3.0 :: Float))
  putStrLn (show (1.1 :: Float))
  putStrLn (show (0.01 :: Double))
