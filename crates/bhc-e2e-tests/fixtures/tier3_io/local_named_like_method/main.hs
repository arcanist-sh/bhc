-- A local named like a class method with method-level constraints is just a
-- local: it must not receive the method's dictionaries.
class Walk a where
  query :: Monoid c => (a -> c) -> a -> c

instance Walk Int where
  query f x = f x

splitFrag :: String -> (String, String)
splitFrag s =
  let (query, fragment) = break (== '#') s
   in (query, fragment)

main :: IO ()
main = do
  let (q, f) = splitFrag "page?x=1#top"
  putStrLn q
  putStrLn f
  putStrLn (query (\n -> show (n + 1)) (41 :: Int))
