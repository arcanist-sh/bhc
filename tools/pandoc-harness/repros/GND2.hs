{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where
import Control.Monad.Reader

newtype F s a = F (Reader s a)
  deriving (Functor, Applicative, Monad)

instance Semigroup a => Semigroup (F s a) where
  (<>) = liftA2 (<>)

runF' :: F s a -> s -> a
runF' (F r) s = runReader r s

one :: F Int String
one = return "a"

two :: F Int String
two = return "b"

main :: IO ()
main = do
  putStrLn (runF' one 0)
  putStrLn (runF' (one <> two) 0)
  putStrLn (runF' (mconcat [one, two, one]) 0)
