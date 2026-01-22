-- |
-- Module      : BHC.Control.Monad.Trans
-- Description : Monad transformers
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Control.Monad.Trans (
    -- * MonadTrans
    MonadTrans(..),
    
    -- * MonadIO
    MonadIO(..),
) where

import BHC.Prelude

-- | Class of monad transformers.
class MonadTrans t where
    -- | Lift a computation from the inner monad.
    lift :: Monad m => m a -> t m a

-- | Class of monads that can lift IO.
class Monad m => MonadIO m where
    -- | Lift an IO action.
    liftIO :: IO a -> m a

instance MonadIO IO where
    liftIO = id
