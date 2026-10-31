{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
-- In an instance whose context constrains the INNER monad (`Monad m =>`),
-- `<$>` over the instance's own monad uses that monad's Functor, not the one
-- reached from the context's `Monad m` (pandoc's `instance Monad m =>
-- HasQuoteContext ParserState m` with `getQuoteContext = stateQuoteContext <$>
-- getState`).
module Main where

newtype P st m a = P { runP :: st -> m (a, st) }

instance Monad m => Functor (P st m) where
  fmap f (P p) = P (\st -> do
    (a, st') <- p st
    return (f a, st'))

getSt :: Monad m => P st m st
getSt = P (\st -> return (st, st))

data Env = Env Int String

class HasLabel st m where
  getLabel :: P st m String

instance Monad m => HasLabel Env m where
  getLabel = (\(Env _ l) -> l) <$> getSt

main :: IO ()
main = do
  (l, _) <- runP getLabel (Env 1 "ctx")
  putStrLn l
