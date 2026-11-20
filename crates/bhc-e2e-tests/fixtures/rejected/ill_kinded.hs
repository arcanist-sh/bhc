-- Types applied to the wrong number or kind of arguments. Each of these
-- declarations compiled: no kind checking ran (#97).
data W = W (Maybe Maybe)

data T f = T (f Int)

type S = Either Int

class C a where
  m :: a -> Int

instance C Maybe where
  m _ = 1

f :: Maybe -> Int
f _ = 1

g :: Int Int -> Int
g _ = 1

h :: T Int -> Int
h _ = 1

k :: S -> Int
k _ = 1

main :: IO ()
main = print (2 :: Int)
