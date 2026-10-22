-- `throwError`/`catchError` used directly at a newtype monad that derives
-- MonadError (pandoc's PandocIO), from `main` where no transformer stack
-- surrounds the call: the occurrence type alone names the ExceptT layer.
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

import Control.Monad.Except
import Control.Monad.State

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad, MonadError String)

runApp :: App a -> IO (Either String a)
runApp (App m) = evalStateT (runExceptT m) 0

main :: IO ()
main = do
  r1 <- runApp (catchError (return "ok") (\_ -> return "caught"))
  print r1
  r2 <- runApp (catchError (throwError "boom") (\e -> return ("caught " ++ e)))
  print r2
  r3 <- runApp (throwError "fail" >> return "after")
  print r3
