-- A multi-parameter class's default method, written through Identity the
-- way pandoc-types' default `walk` is: dispatch at the call site, and a
-- value-position `return` at the library Identity monad.
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
import Data.Functor.Identity (Identity (..), runIdentity)

data I = S String | Sp

class W a b where
  walk :: (a -> a) -> b -> b
  walk f = runIdentity . walkM (return . f)
  walkM :: (Monad m) => (a -> m a) -> b -> m b

instance W I I where
  walkM f x = f x

up :: I -> I
up (S s) = S (s ++ "!")
up x = x

name :: I -> String
name (S s) = s
name Sp = "_"

main :: IO ()
main = do
  putStrLn (name (walk up (S "a")))
  putStrLn (name (runIdentity (walkM (return . up) (S "b"))))
  putStrLn (name (walk up Sp))
