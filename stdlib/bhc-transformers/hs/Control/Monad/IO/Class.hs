-- |
-- Module      : Control.Monad.IO.Class
-- Description : Monads in which IO computations may be embedded
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005).
module Control.Monad.IO.Class
  ( MonadIO (..)
  ) where

-- | Monads that can run an 'IO' action.
class Monad m => MonadIO m where
  liftIO :: IO a -> m a

instance MonadIO IO where
  liftIO io = io
