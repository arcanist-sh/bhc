-- System.IO.Error's tryIOError / catchIOError (previously runtime stubs).
-- `liftIOError` is pandoc's Text.Pandoc.Class.IO helper, behind every
-- PandocIO file method (fileExists, readFileStrict, getDataFileName).
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

import Control.Monad.Except
import Control.Monad.State
import System.Directory (doesFileExist)
import System.IO.Error (catchIOError, tryIOError)

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad, MonadIO, MonadError String)

liftIOError :: (MonadError String m, MonadIO m) => (String -> IO a) -> String -> m a
liftIOError f u = do
  res <- liftIO $ tryIOError $ f u
  case res of
    Left _ -> throwError ("io error: " ++ u)
    Right r -> return r

runApp :: App a -> IO (Either String a)
runApp (App m) = evalStateT (runExceptT m) 0

main :: IO ()
main = do
  r1 <- tryIOError (return (41 + 1 :: Int))
  case r1 of { Left _ -> putStrLn "err"; Right n -> print n }
  r2 <- tryIOError (readFile "/nonexistent/bhc-fixture")
  case r2 of { Left _ -> putStrLn "caught missing file"; Right s -> putStrLn s }
  r3 <- tryIOError (doesFileExist "/nonexistent/bhc-fixture")
  case r3 of { Left _ -> putStrLn "err"; Right True -> putStrLn "exists"; Right False -> putStrLn "absent" }
  s <- catchIOError (readFile "/nonexistent/bhc-fixture") (\_ -> return "fallback")
  putStrLn s
  a1 <- runApp (liftIOError doesFileExist "/nonexistent/bhc-fixture")
  case a1 of { Left e -> putStrLn e; Right b -> putStrLn (if b then "exists" else "absent") }
  a2 <- runApp (liftIOError readFile "/nonexistent/bhc-fixture")
  case a2 of { Left e -> putStrLn e; Right t -> putStrLn t }
