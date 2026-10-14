{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module AppM (App(..), runApp) where

import Control.Monad.State (StateT, runStateT)
import Control.Monad.Except (ExceptT, runExceptT)

-- A monad written as a GND newtype over a transformer stack, the shape of
-- pandoc's `PandocIO`, defined in a DIFFERENT module from its use.
newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad)

runApp :: App a -> IO (Either String a)
runApp m = fmap fst (runStateT (runExceptT (unApp m)) 0)
