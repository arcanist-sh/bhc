{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
-- In an instance whose context constrains the INNER monad (`Monad m =>`), a
-- do-block in the instance's own monad binds with THAT monad's `>>=`, not
-- with `m`'s (pandoc's `withQuoteContext` in `instance Monad m =>
-- HasQuoteContext ParserState m`).
module Main where

newtype P st m a = P { runP :: st -> m (a, st) }

instance Monad m => Functor (P st m) where
  fmap f (P p) = P (\st -> do
    (a, st') <- p st
    return (f a, st'))

instance Monad m => Applicative (P st m) where
  pure a = P (\st -> return (a, st))
  P pf <*> P pa = P (\st -> do
    (f, st1) <- pf st
    (a, st2) <- pa st1
    return (f a, st2))

instance Monad m => Monad (P st m) where
  P p >>= k = P (\st -> do
    (a, st') <- p st
    runP (k a) st')

getSt :: Monad m => P st m st
getSt = P (\st -> return (st, st))

putSt :: Monad m => st -> P st m ()
putSt st = P (\_ -> return ((), st))

data Env = Env { envDepth :: Int, envName :: String }

class HasDepth st m where
  getDepth :: P st m Int
  withDepth :: Int -> P st m a -> P st m a

instance Monad m => HasDepth Env m where
  getDepth = do
    st <- getSt
    return (envDepth st)
  withDepth d p = do
    old <- getDepth
    st <- getSt
    putSt st { envDepth = d }
    r <- p
    st' <- getSt
    putSt st' { envDepth = old }
    return r

probe :: Monad m => P Env m [Int]
probe = do
  a <- getDepth
  b <- withDepth 7 getDepth
  c <- getDepth
  return [a, b, c]

main :: IO ()
main = do
  (xs, st) <- runP probe (Env 1 "env")
  print xs
  putStrLn (envName st)
  print (fmap fst (runP probe (Env 2 "maybe") :: Maybe ([Int], Env)))
