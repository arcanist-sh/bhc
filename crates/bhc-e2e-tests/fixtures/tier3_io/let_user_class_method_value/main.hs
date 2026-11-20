-- A local binding with no signature whose value comes from a user class
-- method, used at one instance. It was generalized: the use instantiated a
-- fresh copy while the method stayed unresolved, and an address was printed.
class Def a where
  def :: a
  name :: a -> String

newtype V = V Int
newtype W = W String

instance Def V where
  def = V 7
  name (V n) = "V" ++ show n

instance Def W where
  def = W "w"
  name (W s) = "W" ++ s

pick :: a -> a -> a
pick x _ = x

useV :: V -> String
useV = name

useW :: W -> String
useW = name

f1 :: String
f1 = let c = pick def def in useV c

f2 :: Int -> String
f2 n = let a = def
           c = pick a def
       in if n > 0 then useW c else useW a

f3 :: IO String
f3 = do
  let a = def
  let c = pick def a
  return (useV c)

main :: IO ()
main = do
  putStrLn f1
  putStrLn (f2 1)
  f3 >>= putStrLn
