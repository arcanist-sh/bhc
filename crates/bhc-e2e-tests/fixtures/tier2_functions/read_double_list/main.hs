-- `read` at Double, [Int] and [Double] (#29).
main :: IO ()
main = do
  let d = read "3.25" :: Double
  print (d * 2)
  print (read " -1.5e2 " :: Double)
  print (read "4" :: Double)
  let xs = read "[1, 2, 3]" :: [Int]
  print (sum xs)
  print xs
  print (read "[]" :: [Int])
  print (read "[0.5,1.5]" :: [Double])
  print (read "42" :: Int)
