-- `liftIO` at ExceptT over StateT over IO (the shape of pandoc's PandocIO).
-- The lifted action must be lowered at the IO base, not at StateT: before,
-- `liftIO (putStrLn "x" >> return 7)` printed "x" and then crashed. In the
-- newtype `App`, the IO action's own `>>` was dispatched at App's Monad
-- dictionary (the enclosing do-block's monad) and `app` returned garbage.
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

import Control.Monad.Except
import Control.Monad.State

type Plain = ExceptT String (StateT Int IO)

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad, MonadIO, MonadError String)

plain :: Plain Int
plain = liftIO (putStrLn "plain" >> return 7)

stateful :: Plain Int
stateful = do
  lift (modify (+ 10))
  n <- liftIO (putStrLn "stateful" >> return 5)
  s <- lift get
  return (n + s)

app :: App Int
app = do
  n <- liftIO (putStrLn "app" >> return 40)
  m <- liftIO (return 2)
  return (n + m)

failing :: Plain Int
failing = do
  _ <- liftIO (putStrLn "before throw" >> return ())
  throwError "boom"

main :: IO ()
main = do
  r1 <- evalStateT (runExceptT plain) 0
  print r1
  r2 <- evalStateT (runExceptT stateful) 1
  print r2
  r4 <- evalStateT (runExceptT failing) 0
  print r4
  r3 <- evalStateT (runExceptT (unApp app)) 1
  print r3
