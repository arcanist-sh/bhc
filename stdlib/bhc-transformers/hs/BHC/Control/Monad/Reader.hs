-- |
-- Module      : BHC.Control.Monad.Reader
-- Description : Reader monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Control.Monad.Reader (
    -- * The Reader monad
    Reader,
    runReader,
    mapReader,
    withReader,
    
    -- * The ReaderT monad transformer
    ReaderT(..),
    runReaderT,
    mapReaderT,
    withReaderT,
    
    -- * Reader operations
    ask,
    local,
    asks,
    reader,
    
    -- * MonadReader class
    MonadReader(..),
) where

import BHC.Prelude
import BHC.Control.Monad.Trans
import BHC.Control.Monad.Identity (Identity(..))

-- | The parameterized reader monad.
type Reader r = ReaderT r Identity

-- | Run a Reader.
runReader :: Reader r a -> r -> a
runReader m = runIdentity . runReaderT m

-- | Map the return value.
mapReader :: (a -> b) -> Reader r a -> Reader r b
mapReader f = mapReaderT (Identity . f . runIdentity)

-- | Transform the environment.
withReader :: (r' -> r) -> Reader r a -> Reader r' a
withReader = withReaderT

-- | The reader monad transformer.
newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }

instance Functor m => Functor (ReaderT r m) where
    fmap f (ReaderT g) = ReaderT (fmap f . g)

instance Applicative m => Applicative (ReaderT r m) where
    pure = ReaderT . const . pure
    ReaderT f <*> ReaderT x = ReaderT $ \r -> f r <*> x r

instance Monad m => Monad (ReaderT r m) where
    ReaderT m >>= k = ReaderT $ \r -> do
        a <- m r
        runReaderT (k a) r

instance MonadTrans (ReaderT r) where
    lift = ReaderT . const

instance MonadIO m => MonadIO (ReaderT r m) where
    liftIO = lift . liftIO

-- | Map the inner computation.
mapReaderT :: (m a -> n b) -> ReaderT r m a -> ReaderT r n b
mapReaderT f (ReaderT g) = ReaderT (f . g)

-- | Transform the environment.
withReaderT :: (r' -> r) -> ReaderT r m a -> ReaderT r' m a
withReaderT f (ReaderT g) = ReaderT (g . f)

-- | Fetch the environment.
ask :: Monad m => ReaderT r m r
ask = ReaderT return

-- | Run with a modified environment.
local :: (r -> r) -> ReaderT r m a -> ReaderT r m a
local f (ReaderT g) = ReaderT (g . f)

-- | Fetch a function of the environment.
asks :: Monad m => (r -> a) -> ReaderT r m a
asks f = ReaderT (return . f)

-- | Create a reader from a function.
reader :: Monad m => (r -> a) -> ReaderT r m a
reader = asks

-- | Class for monads with a readable environment.
class Monad m => MonadReader r m | m -> r where
    askM :: m r
    localM :: (r -> r) -> m a -> m a
    readerM :: (r -> a) -> m a
    
    readerM f = askM >>= return . f

instance Monad m => MonadReader r (ReaderT r m) where
    askM = ask
    localM = local
    readerM = reader
