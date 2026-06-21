loopN :: Int -> Int
loopN 0 = 999
loopN n = loopN (n - 1)

count :: Int -> Int -> Int
count 0 acc = acc
count n acc = count (n - 1) (acc + 1)

main :: IO ()
main = do
  print (loopN 100000)
  print (count 100000 0)
