module Main where

import Control.Monad.State (StateT, evalStateT, get, put)

-- A monad written through a type synonym. Codegen derives a binding's
-- transformer layer from its TYPE, and the synonym hid StateT behind its own
-- constructor: no layer was detected, `return` compiled at the ambient IO
-- layer where it is identity, and `evalStateT` was handed a raw value.
--
-- `viaBind` already worked, because `get`/`put` identify the layer from the
-- body; only the bare `return` form was broken.
type S = StateT Int IO

bare :: S Int
bare = return 42

viaBind :: S Int
viaBind = do
  x <- get
  put (x + 1)
  return (x + 41)

main :: IO ()
main = do
  a <- evalStateT bare 0
  print a
  b <- evalStateT viaBind 1
  print b
