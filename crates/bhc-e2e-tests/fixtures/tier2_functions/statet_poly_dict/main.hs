{-# LANGUAGE GeneralizedNewtypeDeriving, FlexibleInstances #-}
-- `StateT s m` code that reaches its monad through DICTIONARIES inside a
-- `MyMonad m =>` function run at a newtype over ExceptT-over-StateT: a
-- two-variable `Monad m =>` helper (pandoc's `metaToContext`) and a class
-- method from `instance MyMonad m => MyMonad (StateT s m)` (pandoc's
-- `translateTerm` via `PandocMonad (StateT st m)`).
module Main where
import Control.Monad.State
import Control.Monad.Except

class Monad m => MyMonad m where
  tag :: m Int

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad)

instance MyMonad App where
  tag = return 9

instance MyMonad m => MyMonad (StateT s m) where
  tag = lift tag

runApp :: App a -> IO (Either String a)
runApp a = evalStateT (runExceptT (unApp a)) 0

-- two type variables: not specialized; runs on the Monad dictionary
both :: (Monad m, Show a) => (Int -> m a) -> m String
both f = do
  x <- f 1
  y <- f 2
  return (show x ++ show y)

viaDict :: MyMonad m => StateT Int m String
viaDict = do
  modify (+ 5)
  s <- both (\i -> do { st <- get; return (i + st) })
  return s

viaInst :: MyMonad m => StateT Int m Int
viaInst = do
  t <- tag
  return (t + 1)

go1 :: MyMonad m => m String
go1 = evalStateT viaDict 0

go2 :: MyMonad m => m Int
go2 = evalStateT viaInst 0

main :: IO ()
main = do
  r <- runApp go1
  putStrLn (either id id r)
  r2 <- runApp go2
  putStrLn (either id show r2)
