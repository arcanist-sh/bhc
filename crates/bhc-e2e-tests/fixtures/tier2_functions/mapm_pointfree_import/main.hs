-- `mapM`/`forM` brought in by `import Control.Monad` keep their real types,
-- so a point-free `fmap sequence . mapM f` runs in the monad `f` returns.
-- Typed as fresh variables, pandoc's pipe-table reader
-- (`mapM (fmap sequence . mapM cellContents) rows`) ran the list builtin on
-- parsers.
module Main where
import Control.Monad

newtype St a = St { runSt :: Int -> (a, Int) }

instance Functor St where
  fmap f (St g) = St (\s -> let (a, s') = g s in (f a, s'))

instance Applicative St where
  pure a = St (\s -> (a, s))
  St f <*> St g = St (\s -> let (h, s1) = f s; (a, s2) = g s1 in (h a, s2))

instance Monad St where
  St g >>= k = St (\s -> let (a, s1) = g s in runSt (k a) s1)

tick :: Int -> St (Maybe Int)
tick x = St (\s -> (Just (x + s), s + 1))

rows :: [[Int]] -> St [Maybe [Int]]
rows = mapM (fmap sequence . mapM tick)

main :: IO ()
main = do
  let (rs, n) = runSt (rows [[1, 2], [3]]) 10
  print (map (maybe 0 sum) rs)
  print n
  let (xs, _) = runSt (forM [5, 6] tick) 0
  print (map (maybe 0 id) xs)
