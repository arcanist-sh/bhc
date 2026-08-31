module Main where

import Control.Monad (ap)

-- `(<*>) = ap` is how parsec writes its Applicative instance; codegen has no
-- implementation for `ap` and aborted with `stub: ap not implemented`.
newtype Box a = Box { runBox :: [a] }

instance Functor Box where
  fmap f (Box xs) = Box (map f xs)

instance Applicative Box where
  pure x = Box [x]
  (<*>) = ap

instance Monad Box where
  Box xs >>= k = Box (concatMap (runBox . k) xs)

main :: IO ()
main = do
  print (runBox (Box [(+ 1), (* 2)] <*> Box [10, 20]))
  print (runBox (Box [1, 2] >>= \x -> Box [x * 10, x * 10 + 1]))
