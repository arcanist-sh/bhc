{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
-- A two-parameter class whose method returns a different monad that wraps
-- both class arguments (pandoc's `HasQuoteContext st m` with
-- `getQuoteContext :: ParsecT s st m QuoteContext`): a constrained function
-- selects the method from the dictionary in scope, and a function with only
-- `Monad m` in scope builds it from an instance whose monad parameter is a
-- variable (`instance Monad m => HasQuoteContext ParserState m`).
module Main where

newtype P st m a = P { runP :: st -> m (a, st) }

class HasDepth st m where
  getDepth :: P st m Int

data Env = Env Int

instance HasDepth Env IO where
  getDepth = P (\st@(Env d) -> return (d, st))

class Scoped st m where
  withDepth :: Int -> P st m a -> P st m a

data Ctx = Ctx Int

instance Monad m => Scoped Ctx m where
  withDepth d (P p) = P (\_ -> p (Ctx d))

depthOf :: Monad m => P Ctx m Int
depthOf = P (\st@(Ctx d) -> return (d, st))

nested :: Monad m => P Ctx m Int
nested = withDepth 7 depthOf

atLeast :: (HasDepth st m, Monad m) => Int -> P st m Bool
atLeast n = P (\st -> do
  (d, st') <- runP getDepth st
  return (d >= n, st'))

main :: IO ()
main = do
  (a, _) <- runP (atLeast 2) (Env 3)
  print a
  (b, _) <- runP (atLeast 5) (Env 3)
  print b
  (c, _) <- runP nested (Ctx 1)
  print c
