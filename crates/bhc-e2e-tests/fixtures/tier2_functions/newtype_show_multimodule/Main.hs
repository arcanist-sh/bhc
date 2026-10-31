-- Derived Show of newtypes (record syntax, precedence) and of types
-- imported from another module (#52).
import Types

newtype L = L { unL :: Int } deriving Show

main :: IO ()
main = do
  print (W 3)
  print (S (2 :: Int))
  print (P 3, P (-3))
  print (R 1 True)
  print [W 1, W 2]
  print (Just (W 5), Just (P 4))
  print (L 7)
  putStrLn (show (W 9))
