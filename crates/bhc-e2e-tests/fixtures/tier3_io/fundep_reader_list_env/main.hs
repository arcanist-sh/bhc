{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies, FlexibleInstances,
             TupleSections #-}
-- `asks fst` in a reader whose environment holds a list gets its
-- `MonadReader` dictionary even when the list flows through a generic `<$>`
-- (pandoc's AnnotatedTable: `colspec <- asks fst; let hang = (1, ) <$>
-- colspec`). The occurrence's resolved type spelled the list `[] Int` and the
-- argument `fst` left `r` open, so no instance matched and `asks` ran with
-- no dictionary.
module Main where

newtype Id a = Id { runId :: a }

instance Functor Id where
  fmap f (Id a) = Id (f a)
instance Applicative Id where
  pure = Id
  Id f <*> Id a = Id (f a)
instance Monad Id where
  Id a >>= k = k a

newtype RS r s m a = RS { runRS :: r -> s -> m (a, s) }

instance Functor m => Functor (RS r s m) where
  fmap f m = RS (\r s -> fmap (\(a, s') -> (f a, s')) (runRS m r s))
instance Monad m => Applicative (RS r s m) where
  pure a = RS (\_ s -> return (a, s))
  RS mf <*> RS mx = RS (\r s -> mf r s >>= \(f, s1) ->
    mx r s1 >>= \(x, s2) -> return (f x, s2))
instance Monad m => Monad (RS r s m) where
  m >>= k = RS (\r s -> runRS m r s >>= \(a, s1) -> runRS (k a) r s1)

class Monad m => MonadReader r m | m -> r where
  ask :: m r

instance Monad m => MonadReader r (RS r s m) where
  ask = RS (\r s -> return (r, s))

asks :: MonadReader r m => (r -> a) -> m a
asks f = ask >>= \r -> return (f r)

getS :: Monad m => RS r s m s
getS = RS (\_ s -> return (s, s))

putS :: Monad m => s -> RS r s m ()
putS s = RS (\_ _ -> return ((), s))

type M a = RS ([Int], Int) Int Id a

section :: [[Int]] -> M [Int]
section rows = do
  cs <- asks fst
  let hang = (1, ) <$> cs
  go hang id rows
 where
  go h acc (r : rs) = do
    n <- getS
    putS (n + 1)
    go h (acc . ((n * 100 + sum r + length h) :)) rs
  go _ acc [] = return $ acc []

readEnv :: M Int
readEnv = do
  cs <- asks fst
  return (length cs)

main :: IO ()
main = do
  print (fst (runId (runRS readEnv ([7, 8], 2) 0)))
  print (fst (runId (runRS (section [[1, 2], [3]]) ([7, 8], 2) 0)))
  print (snd (runId (runRS (section [[1, 2], [3]]) ([7, 8], 2) 0)))
