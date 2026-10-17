{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
-- |
-- Module      : Control.Monad.Error.Class
-- Description : The MonadError class
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.Error.Class
  ( MonadError (..)
  , liftEither
  ) where

import Control.Monad.Trans.Class (lift)
import qualified Control.Monad.Trans.Except as E
import Control.Monad.Trans.Maybe (MaybeT (..))
import Control.Monad.Trans.Reader (ReaderT (..))
import Control.Monad.Trans.State.Strict (StateT (..))
import Control.Monad.Trans.Writer.Strict (WriterT (..))

-- | Monads that can fail with an error of type @e@ and handle it.
class Monad m => MonadError e m | m -> e where
  -- | Fail with an error.
  throwError :: e -> m a
  -- | Handle an error.
  catchError :: m a -> (e -> m a) -> m a

-- | Lift an 'Either' into the monad.
liftEither :: MonadError e m => Either e a -> m a
liftEither ea = case ea of
  Left e -> throwError e
  Right a -> return a

instance MonadError e (Either e) where
  throwError = Left
  catchError ea h = case ea of
    Left e -> h e
    Right a -> Right a

instance Monad m => MonadError e (E.ExceptT e m) where
  throwError = E.throwE
  catchError = E.catchE

instance MonadError e m => MonadError e (StateT s m) where
  throwError e = lift (throwError e)
  catchError m h = StateT (\s -> catchError (runStateT m s) (\e -> runStateT (h e) s))

instance MonadError e m => MonadError e (ReaderT r m) where
  throwError e = lift (throwError e)
  catchError m h = ReaderT (\r -> catchError (runReaderT m r) (\e -> runReaderT (h e) r))

instance (Monoid w, MonadError e m) => MonadError e (WriterT w m) where
  throwError e = lift (throwError e)
  catchError m h = WriterT (catchError (runWriterT m) (\e -> runWriterT (h e)))

instance MonadError e m => MonadError e (MaybeT m) where
  throwError e = lift (throwError e)
  catchError m h = MaybeT (catchError (runMaybeT m) (\e -> runMaybeT (h e)))
