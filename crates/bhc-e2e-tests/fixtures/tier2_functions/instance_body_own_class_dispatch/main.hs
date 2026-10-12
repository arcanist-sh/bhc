-- Inside `instance MonadLog m => MonadLog (R m)`, class methods used at
-- `R m` (a default method, the do-block's bind) must come from the instance
-- being defined, not from the constraint dictionary for `m` (pandoc's
-- `trace` in the ParsecT instance of PandocMonad).
module Main where
import Control.Monad (when)

class Monad m => MonadLog m where
  getLvl :: m Int
  getsLvl :: (Int -> a) -> m a
  getsLvl f = f <$> getLvl
  logIt :: String -> m ()
  logIt _ = return ()
  logOut :: String -> m ()

newtype R m a = R { runR :: Int -> m a }

instance Functor m => Functor (R m) where
  fmap f (R g) = R (\e -> fmap f (g e))
instance Monad m => Applicative (R m) where
  pure x = R (\_ -> return x)
  R f <*> R g = R (\e -> f e >>= \h -> g e >>= \x -> return (h x))
instance Monad m => Monad (R m) where
  R g >>= k = R (\e -> g e >>= \x -> runR (k x) e)

lift' :: m a -> R m a
lift' m = R (const m)

ask' :: Monad m => R m Int
ask' = R return

instance MonadLog IO where
  getLvl = return 3
  logOut = putStrLn

instance MonadLog m => MonadLog (R m) where
  getLvl = lift' getLvl
  logOut s = lift' (logOut s)
  logIt s = do
    t <- getsLvl (> 1)
    when t $ do
      n <- ask'
      lift' (logOut (show n ++ s))

main :: IO ()
main = do
  runR (logIt "x") 7
  a <- runR getLvl 0
  print (a + 0 :: Int)
  c <- getsLvl (* 10)
  print (c + 0 :: Int)
