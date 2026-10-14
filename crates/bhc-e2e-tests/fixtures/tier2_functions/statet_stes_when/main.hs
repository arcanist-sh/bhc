{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- `when`/`unless` inside `StateT s` over a newtype of ExceptT-over-StateT:
-- each is a COMPUTATION (`if c then act else return ()`), sequenced by `>>`.
module Main where
import Control.Monad.State
import Control.Monad.Except

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad)

runApp :: App a -> IO (Either String a)
runApp a = evalStateT (runExceptT (unApp a)) 0

bump :: Bool -> StateT Int App ()
bump b = when b (modify (+ 1))

steps :: StateT Int App Int
steps = do
  bump True
  bump False
  unless False (modify (* 10))
  unless True (modify (+ 1000))
  when True (lift (return ()))
  get

report :: App Int -> IO ()
report act = do
  r <- runApp act
  case r of
    Left e -> putStrLn ("Left " ++ e)
    Right n -> print n

main :: IO ()
main = do
  report (evalStateT steps 4)
  report (evalStateT (when False (modify (+ 1)) >> get) 7)
