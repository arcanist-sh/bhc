{-# LANGUAGE TupleSections #-}
-- A tuple section is a function: nothing to print (#106).
import Control.Monad (when)

main :: IO ()
main = do
  let i = 3 :: Int
  when (odd i) (print (i,))
