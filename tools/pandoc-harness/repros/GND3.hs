{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where
import Control.Monad.Reader

newtype F s a = F (Reader s a)
  deriving (Functor, Applicative, Monad)

-- Eta-EXPANDED, unlike pandoc's point-free `(<>) = liftA2 (<>)`.
instance Semigroup a => Semigroup (F s a) where
  a <> b = liftA2 (<>) a b

runF' :: F s a -> s -> a
runF' (F r) s = runReader r s

one :: F Int String
one = return "a"

two :: F Int String
two = return "b"

main :: IO ()
main = do
  putStrLn (runF' (liftA2 (<>) one two) 0)
  putStrLn (runF' (one <> two) 0)
