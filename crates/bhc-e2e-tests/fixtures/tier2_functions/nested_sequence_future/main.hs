-- The `sequence` inside `fmap sequence . mapM f` runs in the monad of the
-- values it collects (here a reader-like `Fut`), not in the enclosing
-- do-block's. Re-exported by `import Control.Monad`, it was typed as a
-- fresh variable and expanded at the enclosing `St`, so every row of
-- pandoc's pipe tables came back empty or broken.
{-# LANGUAGE GeneralizedNewtypeDeriving, ScopedTypeVariables #-}
module Main where
import Control.Monad

newtype Rd s a = Rd { runRd :: s -> a }

instance Functor (Rd s) where
  fmap f (Rd g) = Rd (f . g)
instance Applicative (Rd s) where
  pure x = Rd (const x)
  Rd f <*> Rd g = Rd (\s -> f s (g s))
instance Monad (Rd s) where
  Rd g >>= k = Rd (\s -> runRd (k (g s)) s)

newtype Fut a = Fut { runDelayed :: Rd Int a }
  deriving (Functor, Applicative, Monad)

runF :: Fut a -> Int -> a
runF f = runRd (runDelayed f)

newtype St a = St { runSt :: Int -> (a, Int) }
instance Functor St where
  fmap f (St g) = St (\s -> let (a, s') = g s in (f a, s'))
instance Applicative St where
  pure a = St (\s -> (a, s))
  St f <*> St g = St (\s -> let { (h, s1) = f s; (a, s2) = g s1 } in (h a, s2))
instance Monad St where
  St g >>= k = St (\s -> let (a, s1) = g s in runSt (k a) s1)

cellP :: String -> St (Fut [String])
cellP t = St (\s -> (return [t ++ show s], s + 1))

table :: [String] -> [[String]] -> St (Int, [[[String]]])
table heads rows = do
  (hc :: Fut [[String]]) <- sequence <$> mapM cellP heads
  (rs :: Fut [[[String]]]) <- sequence <$> mapM (fmap sequence . mapM cellP) rows
  return (length (runF hc 0), runF rs 0)

main :: IO ()
main = do
  let ((n, rs), _) = runSt (table ["a", "b"] [["1", "2"], ["3", "4"]]) 0
  print n
  print (length rs)
  print (map length rs)
  print (concat (concat rs))
