-- Enum and Bounded at the standard types, and fractional values whose type
-- is defaulted. Each line printed something else: `fromEnum True` and
-- `map fromEnum [LT, GT]` gave addresses, `pred GT` was GT, `[LT ..]` and
-- `[minBound .. maxBound :: Bool]` printed nothing, `minBound :: Bool`
-- was Int's bound, `succ` at Integer and `recip` crashed, and `2 ** 3` in
-- a tuple showed the Double's bits as an Int.
data Color = Red | Green | Blue deriving (Show, Eq, Enum, Bounded)

data Size = Small | Large deriving (Show, Enum, Bounded)

main :: IO ()
main = do
  print (fromEnum True, fromEnum (), fromEnum GT, fromEnum 'A')
  print (map fromEnum [LT, EQ, GT], map fromEnum [Red, Blue])
  print (succ False, pred GT, succ LT, pred True)
  print ([LT ..], [False ..], [GT, EQ ..], [False .. True])
  print (minBound :: Bool, maxBound :: Ordering, fromEnum (maxBound :: Char))
  print ([minBound .. maxBound :: Bool], [minBound ..] :: [Ordering])
  print (minBound :: Color, maxBound :: Size, [minBound .. maxBound :: Size])
  print (succ (2 :: Integer), pred (10 ^ 20 :: Integer), map succ [1, 2 :: Integer])
  let n = 10 :: Int
  print (fromIntegral n / 4, 2 ** 3, n)
  print (recip 4, map recip [1, 2, 4 :: Double])
  print [(1 / 2, 'x')]
  print (Just (2 ** 10))
