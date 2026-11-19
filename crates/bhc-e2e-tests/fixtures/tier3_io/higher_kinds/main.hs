-- Higher-kinded declarations the kind checker must accept.
data Proxy a = Proxy

data Wrap f a = Wrap (f a)

newtype Compose f g a = Compose (f (g a))

data Rose a = Rose a [Rose a]

type Pairs = Either (Int, Int)

class Container f where
  empty :: f a
  insert :: a -> f a -> f a
  toL :: f a -> [a]

newtype Box a = Box [a]

instance Container Box where
  empty = Box []
  insert x (Box xs) = Box (x : xs)
  toL (Box xs) = xs

unwrap :: Wrap Maybe Int -> Maybe Int
unwrap (Wrap m) = m

proxies :: (Proxy Maybe, Proxy Int, Proxy Either)
proxies = (Proxy, Proxy, Proxy)

pairs :: Pairs String
pairs = Right "ok"

size :: Rose a -> Int
size (Rose _ kids) = 1 + sum (map size kids)

decompose :: Compose Maybe [] Int -> Maybe [Int]
decompose (Compose x) = x

main :: IO ()
main = do
  print (unwrap (Wrap (Just 3)))
  print (toL (insert (1 :: Int) (insert 2 empty) :: Box Int))
  print (size (Rose 'a' [Rose 'b' [], Rose 'c' [Rose 'd' []]]))
  print (decompose (Compose (Just [1, 2])))
  case pairs of
    Right s -> putStrLn s
    Left _ -> putStrLn "pair"
  case proxies of
    (Proxy, Proxy, Proxy) -> putStrLn "proxies"
