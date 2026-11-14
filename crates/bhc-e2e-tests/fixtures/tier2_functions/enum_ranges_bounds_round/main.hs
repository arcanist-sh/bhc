-- Ranges over a derived Enum, builtin bounds beside a user Bounded type, and
-- `round`. `[Wed ..]` was empty, `[Mon, Wed .. Sat]` gave `[Mon]` and
-- `[Mon, Wed ..]` crashed; with one user Bounded type, `maxBound :: Int` was
-- that type's last constructor; `round 2.5` was 3 (Haskell rounds half to
-- even).
data Day = Mon | Tue | Wed | Thu | Fri | Sat | Sun
  deriving (Show, Eq, Ord, Enum, Bounded)

main :: IO ()
main = do
  print [Wed ..]
  print [Mon .. Thu]
  print [Mon, Wed .. Sat]
  print [Mon, Wed ..]
  print [Sun, Fri ..]
  print [Sat, Fri .. Tue]
  print [minBound .. maxBound :: Day]
  print (succ Mon, pred Sun, fromEnum Wed, toEnum 4 :: Day)
  print (maxBound :: Int, minBound :: Int)
  print (fromIntegral (maxBound :: Int) + (1 :: Integer))
  print (fromEnum (maxBound :: Char))
  print (map round [0.5, 1.5, 2.5, 3.5, -2.5 :: Double] :: [Int])
