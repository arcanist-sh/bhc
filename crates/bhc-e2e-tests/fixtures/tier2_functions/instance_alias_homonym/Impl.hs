-- Plain functions named like the `Store` class's methods, constrained by the
-- class itself (pandoc's Text.Pandoc.Class.IO next to PandocMonad).
module Impl (fetch, label) where

import Control.Monad.IO.Class (MonadIO, liftIO)
import Cls (Store)
import qualified Cls

fetch :: (Store m, MonadIO m) => Int -> m Int
fetch n = do
  b <- Cls.base
  liftIO (return (n * 2 + b))

label :: MonadIO m => String -> m String
label s = liftIO (return ("<" ++ s ++ ">"))
