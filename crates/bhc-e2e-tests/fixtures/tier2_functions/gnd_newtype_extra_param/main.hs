-- A newtype with a parameter BEFORE the one the class abstracts over —
-- pandoc's `Future s a` over `Reader s a`. Three things have to hold:
--   * the derived Functor/Applicative/Monad instances exist at all (they were
--     declined for anything but a single-parameter newtype, so the instance
--     came out empty);
--   * they are keyed and named at the applied head `F s`, or an importing
--     module calls a symbol the producer never defined;
--   * the derived `fmap` DELEGATES to the inner functor instead of applying
--     the function to the whole wrapped action.
-- `liftA2` comes along for the ride: bhc has no implementation of its own, and
-- it is how pandoc's `Semigroup (Future s a)` is written.
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

import Control.Monad.Reader

newtype F s a = F (Reader s a)
  deriving (Functor, Applicative, Monad)

runF' :: F s a -> s -> a
runF' (F r) s = runReader r s

prog :: F Int Int
prog = do
  a <- return 1
  b <- return 2
  return (a + b)

oneS :: F Int String
oneS = return "a"

twoS :: F Int String
twoS = return "b"

main :: IO ()
main = do
  print (runF' prog 0)
  print (runF' (fmap (* 2) prog) 0)
  print (runF' (liftA2 (+) prog prog) 0)
  putStrLn (runF' (liftA2 (++) oneS twoS) 0)
