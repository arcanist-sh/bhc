-- |
-- Module      : Control.Monad.Trans
-- Description : MonadTrans and MonadIO
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.Trans
  ( MonadTrans (..)
  , MonadIO (..)
  ) where

import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (MonadTrans (..))
