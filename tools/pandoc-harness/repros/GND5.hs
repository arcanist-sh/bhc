{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where
import Control.Monad.Reader

newtype F s a = F (Reader s a)
  deriving (Functor, Applicative, Monad)

runF' :: F s a -> s -> a
runF' (F r) s = runReader r s

oneS :: F Int String
oneS = return "a"

twoS :: F Int String
twoS = return "b"

handwritten :: F Int String
handwritten = oneS >>= \x -> twoS >>= \y -> return (x ++ y)

main :: IO ()
main = do
  putStrLn (runF' handwritten 0)
  putStrLn (runF' (liftA2 (++) oneS twoS) 0)
