-- |
-- Module      : Control.Monad.Trans.Class
-- Description : The class of monad transformers
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005).
module Control.Monad.Trans.Class
  ( MonadTrans (..)
  ) where

-- | Monad transformers: @lift@ embeds a computation of the inner monad.
--
-- Law: @lift . return = return@, @lift (m >>= f) = lift m >>= (lift . f)@.
class MonadTrans t where
  lift :: Monad m => m a -> t m a
