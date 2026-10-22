-- A function that shares its name with a class method (Impl.fetch vs
-- Store's fetch) must be called as that function, not dispatched as the
-- method: the instance below is written `fetch = Impl.fetch`, as pandoc's
-- PandocIO instance is for every file method, and Impl.fetch itself needs
-- the very instance being defined.
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

import Control.Monad.Except
import Control.Monad.State
import Cls
import qualified Impl

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad, MonadIO)

instance Store App where
  fetch = Impl.fetch
  label = Impl.label
  base = return 0

runApp :: App a -> IO (Either String a)
runApp (App m) = evalStateT (runExceptT m) 0

useStore :: Store m => m (Int, String)
useStore = do
  n <- fetch 20
  s <- label "x"
  return (n + 1, s)

main :: IO ()
main = do
  r1 <- runApp (fetch 21)
  case r1 of { Left e -> putStrLn e; Right n -> print n }
  r2 <- runApp (Impl.label "direct")
  case r2 of { Left e -> putStrLn e; Right s -> putStrLn s }
  r3 <- runApp useStore
  case r3 of { Left e -> putStrLn e; Right (n, s) -> putStrLn (show n ++ " " ++ s) }
