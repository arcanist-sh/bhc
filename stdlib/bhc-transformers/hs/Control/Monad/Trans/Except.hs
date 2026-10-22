-- |
-- Module      : Control.Monad.Trans.Except
-- Description : The exception monad transformer
-- License     : BSD-3-Clause
--
-- Part of BHC's transformers library (spec/BHC-BRIEF-0005). The overloaded
-- operations live in "Control.Monad.Error.Class".
module Control.Monad.Trans.Except
  ( -- * The Except monad
    Except
  , except
  , runExcept
  , mapExcept
  , withExcept
    -- * The ExceptT transformer
  , ExceptT (..)
  , mapExceptT
  , withExceptT
    -- * Exception operations
  , throwE
  , catchE
  ) where

import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (MonadTrans (..))
import Data.Functor.Identity (Identity (..))

-- | A computation that may fail with an exception of type @e@.
newtype ExceptT e m a = ExceptT { runExceptT :: m (Either e a) }

-- | An exception monad over 'Identity'.
type Except e = ExceptT e Identity

instance Functor m => Functor (ExceptT e m) where
  fmap f (ExceptT m) = ExceptT (fmap (either Left (\a -> Right (f a))) m)

instance Monad m => Applicative (ExceptT e m) where
  pure a = ExceptT (return (Right a))
  ExceptT mf <*> ExceptT mx = ExceptT (mf >>= \ef -> case ef of
    Left e -> return (Left e)
    Right f -> mx >>= \ex -> case ex of
      Left e -> return (Left e)
      Right x -> return (Right (f x)))

instance Monad m => Monad (ExceptT e m) where
  ExceptT m >>= k = ExceptT (m >>= \ea -> case ea of
    Left e -> return (Left e)
    Right a -> runExceptT (k a))

instance MonadTrans (ExceptT e) where
  lift m = ExceptT (m >>= \a -> return (Right a))

instance MonadIO m => MonadIO (ExceptT e m) where
  liftIO io = lift (liftIO io)

-- | Lift an 'Either' into the exception monad.
except :: Monad m => Either e a -> ExceptT e m a
except ea = ExceptT (return ea)

-- | Run an 'Except' computation.
runExcept :: Except e a -> Either e a
runExcept m = runIdentity (runExceptT m)

-- | Transform the result of an 'Except' computation.
mapExcept :: (Either e a -> Either e' b) -> Except e a -> Except e' b
mapExcept f = mapExceptT (\m -> Identity (f (runIdentity m)))

-- | Transform the exception.
withExcept :: (e -> e') -> Except e a -> Except e' a
withExcept = withExceptT

-- | Transform the inner action.
mapExceptT :: (m (Either e a) -> n (Either e' b)) -> ExceptT e m a -> ExceptT e' n b
mapExceptT f (ExceptT m) = ExceptT (f m)

-- | Transform the exception.
withExceptT :: Functor m => (e -> e') -> ExceptT e m a -> ExceptT e' m a
withExceptT f (ExceptT m) = ExceptT (fmap (either (\e -> Left (f e)) Right) m)

-- | Fail with an exception.
throwE :: Monad m => e -> ExceptT e m a
throwE e = ExceptT (return (Left e))

-- | Handle an exception.
catchE :: Monad m => ExceptT e m a -> (e -> ExceptT e' m a) -> ExceptT e' m a
catchE (ExceptT m) handler = ExceptT (m >>= \ea -> case ea of
  Left e -> runExceptT (handler e)
  Right a -> return (Right a))
