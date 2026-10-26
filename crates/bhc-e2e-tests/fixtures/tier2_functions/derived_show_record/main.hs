-- Derived Show uses record syntax for record constructors (#49).
data P = P { name :: String, age :: Int } deriving Show
data Q = Q { inner :: P, delta :: Int } deriving Show
data V = V Int | W { w :: Double } deriving Show
main :: IO ()
main = do
  print (P "Ada" 36)
  print (Q (P "Bo" 1) (-2))
  print (Just (P "C" 3))
  print [V 1, W 2.5]
