{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

-- `StateT s M` where `M` is a GND newtype over a transformer stack (the shape
-- of pandoc's `PandocIO`): the inner monad has to be read through the newtype.

import Control.Monad.State (StateT, evalStateT, runStateT, get, modify, lift)
import Control.Monad.Except (ExceptT, runExceptT)

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad)

runApp :: App a -> IO (Either String a)
runApp m = fmap fst (runStateT (runExceptT (unApp m)) 0)

constant :: App Int
constant = evalStateT (return 5) (0 :: Int)

counter :: App Int
counter = evalStateT (do { modify (+ 1); s <- get; return (s * 2) }) (7 :: Int)

collect :: App [Int]
collect = evalStateT (mapM (\x -> do { modify (+ x); get }) [1, 2, 3]) (10 :: Int)

main :: IO ()
main = do
  r1 <- runApp constant
  case r1 of { Left _ -> putStrLn "Left"; Right n -> print n }
  r2 <- runApp counter
  case r2 of { Left _ -> putStrLn "Left"; Right n -> print n }
  r3 <- runApp collect
  case r3 of { Left _ -> putStrLn "Left"; Right ns -> print ns }
