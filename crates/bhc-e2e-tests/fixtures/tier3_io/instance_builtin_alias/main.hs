-- Instance methods defined as a builtin class method (#30).
class Sh a where
  sh :: a -> String

instance Sh Bool where
  sh = show

instance Sh Int where
  sh = show

class Same a where
  same :: a -> a -> Bool

instance Same Int where
  same = (==)

main :: IO ()
main = do
  putStrLn (sh True)
  putStrLn (sh (42 :: Int))
  print (same (3 :: Int) 3, same (3 :: Int) 4)
