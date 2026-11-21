-- The Floating functions at Float, and at Double by default. They were
-- typed at Double only, so `sqrt (2 :: Float)` was a type error; at Float
-- the result is rounded to Float precision and printed as a Float.
main :: IO ()
main = do
  print (sqrt (2 :: Float))
  print (exp (1 :: Float), sin (1 :: Float), asin (0.5 :: Float))
  print (sqrt 2, exp 1, atan2 1 2, logBase 2 1024)
  let x = 2 :: Float
  print (sqrt x * sqrt x, isNaN x, atan2 x 1)
  print (map sqrt [1, 4, 9 :: Float])
  print (sqrt (16 :: Double), map sqrt [2 :: Double])
  let y = sqrt (2 :: Float)
  print y
