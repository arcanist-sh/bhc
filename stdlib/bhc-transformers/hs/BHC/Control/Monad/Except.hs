-- |
-- Module      : BHC.Control.Monad.Except
-- Description : Exception monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Control.Monad.Except (
    -- * The Except monad
    Except,
    runExcept,
    mapExcept,
    withExcept,
    
    -- * The ExceptT monad transformer
    ExceptT(..),
    runExceptT,
    mapExceptT,
    withExceptT,
    
    -- * Exception operations
    throwE,
    catchE,
    
    -- * MonadError class
    MonadError(..),
) where

import BHC.Prelude
import BHC.Control.Monad.Trans
import BHC.Control.Monad.Identity (Identity(..))

-- | The exception monad.
type Except e = ExceptT e Identity

-- | Run an Except.
runExcept :: Except e a -> Either e a
runExcept = runIdentity . runExceptT

-- | Map the result.
mapExcept :: (Either e a -> Either e' b) -> Except e a -> Except e' b
mapExcept f = mapExceptT (Identity . f . runIdentity)

-- | Transform the exception.
withExcept :: (e -> e') -> Except e a -> Except e' a
withExcept = withExceptT

-- | The exception monad transformer.
newtype ExceptT e m a = ExceptT { runExceptT :: m (Either e a) }

instance Functor m => Functor (ExceptT e m) where
    fmap f (ExceptT m) = ExceptT $ fmap (fmap f) m

instance Monad m => Applicative (ExceptT e m) where
    pure = ExceptT . return . Right
    ExceptT mf <*> ExceptT mx = ExceptT $ do
        ef <- mf
        case ef of
            Left e  -> return (Left e)
            Right f -> fmap (fmap f) mx

instance Monad m => Monad (ExceptT e m) where
    ExceptT m >>= k = ExceptT $ do
        ea <- m
        case ea of
            Left e  -> return (Left e)
            Right a -> runExceptT (k a)

instance MonadTrans (ExceptT e) where
    lift = ExceptT . fmap Right

instance MonadIO m => MonadIO (ExceptT e m) where
    liftIO = lift . liftIO

-- | Map the inner computation.
mapExceptT :: (m (Either e a) -> n (Either e' b)) -> ExceptT e m a -> ExceptT e' n b
mapExceptT f (ExceptT m) = ExceptT (f m)

-- | Transform the exception.
withExceptT :: Functor m => (e -> e') -> ExceptT e m a -> ExceptT e' m a
withExceptT f = mapExceptT (fmap (either (Left . f) Right))

-- | Throw an exception.
throwE :: Monad m => e -> ExceptT e m a
throwE = ExceptT . return . Left

-- | Catch an exception.
catchE :: Monad m => ExceptT e m a -> (e -> ExceptT e' m a) -> ExceptT e' m a
catchE (ExceptT m) handler = ExceptT $ do
    ea <- m
    case ea of
        Left e  -> runExceptT (handler e)
        Right a -> return (Right a)

-- | Class for monads with exceptions.
class Monad m => MonadError e m | m -> e where
    throwError :: e -> m a
    catchError :: m a -> (e -> m a) -> m a

instance Monad m => MonadError e (ExceptT e m) where
    throwError = throwE
    catchError = catchE
