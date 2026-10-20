-- Test: State operations in a function generic in its state type
-- Verifies: `modify` inside `HasPos u => State u a` gets its MonadState
-- dictionary — the state type `u` is fixed by `StateT u Identity` through
-- the class's functional dependency (pandoc's parseFromString' is this
-- shape).

import Control.Monad.State.Strict

class HasPos st where
  setPos :: Maybe Int -> st -> st
  getPos :: st -> Maybe Int

data PState = PState {psPos :: Maybe Int}

instance HasPos PState where
  setPos p st = st {psPos = p}
  getPos st = psPos st

withTemporaryPos :: HasPos u => State u (Maybe Int)
withTemporaryPos = do
  old <- getPos <$> get
  modify (setPos (Just 9))
  new <- getPos <$> get
  modify (setPos old)
  return new

current :: State PState (Maybe Int)
current = getPos <$> get

main :: IO ()
main = do
  print (maybe 0 id (evalState current (PState (Just 3))))
  print (maybe 0 id (evalState withTemporaryPos (PState (Just 3))))
  print (maybe 0 id (psPos (execState withTemporaryPos (PState (Just 3)))))
