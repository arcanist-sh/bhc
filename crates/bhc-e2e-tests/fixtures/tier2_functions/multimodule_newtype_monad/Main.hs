module Main where

-- `return` at `StateT Int App` must stay a StateT action, not dispatch to the
-- imported `App`'s own `pure`; and codegen must read `App`'s layer through
-- the newtype even though `App` is defined in another module.

import Control.Monad.State (evalStateT, get, modify)
import AppM

constant :: App Int
constant = evalStateT (return 5) (0 :: Int)

counter :: App Int
counter = evalStateT (do { modify (+ 1); s <- get; return (s * 2) }) (7 :: Int)

main :: IO ()
main = do
  r1 <- runApp constant
  case r1 of { Left _ -> putStrLn "Left"; Right n -> print n }
  r2 <- runApp counter
  case r2 of { Left _ -> putStrLn "Left"; Right n -> print n }
