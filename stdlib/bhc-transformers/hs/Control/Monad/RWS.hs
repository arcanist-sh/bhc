-- |
-- Module      : Control.Monad.RWS
-- Description : Reader-writer-state monads, overloaded
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005). Re-exports the strict
-- variant, as the other monads here do.
module Control.Monad.RWS
  ( module Control.Monad.RWS.Strict
  ) where

import Control.Monad.RWS.Strict
