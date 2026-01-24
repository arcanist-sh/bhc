-- |
-- Module      : BHC.Control.Monad.Except
-- Description : Exception monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The Except monad for computations that may fail with an error.
--
-- = Overview
--
-- The Except monad represents computations that may fail with an
-- error value. Unlike 'Maybe', 'Except' carries information about
-- /why/ the computation failed.
--
-- = Usage
--
-- @
-- import BHC.Control.Monad.Except
--
-- data AppError = NotFound String | InvalidInput String
--
-- lookupUser :: UserId -> Except AppError User
-- lookupUser uid = case findUser uid of
--     Nothing   -> throwE (NotFound $ "User " ++ show uid)
--     Just user -> return user
--
-- -- Handle errors
-- result = runExcept (lookupUser 42) \`catchE\` \\err ->
--     case err of
--         NotFound msg -> return defaultUser
--         _            -> throwE err
-- @
--
-- = Transformer Usage
--
-- @
-- type App = ExceptT AppError IO
--
-- loadConfig :: FilePath -> App Config
-- loadConfig path = do
--     exists <- liftIO (doesFileExist path)
--     unless exists $ throwE (NotFound path)
--     liftIO (readConfig path)
-- @

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

-- | /O(1)/. Signal an error, aborting the computation.
--
-- ==== __Examples__
--
-- >>> runExcept (throwE "error" :: Except String Int)
-- Left "error"
--
-- @
-- validate :: Int -> Except String Int
-- validate n
--     | n < 0     = throwE "negative number"
--     | otherwise = return n
-- @
throwE :: Monad m => e -> ExceptT e m a
throwE = ExceptT . return . Left

-- | /O(1)/. Handle an error by running a recovery computation.
--
-- If the first computation succeeds, its result is returned.
-- If it fails, the handler is called with the error value.
--
-- ==== __Examples__
--
-- >>> runExcept (throwE "oops" `catchE` \_ -> return 42)
-- Right 42
--
-- @
-- withDefault :: a -> Except e a -> Except e a
-- withDefault def m = m \`catchE\` \\_ -> return def
-- @
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
