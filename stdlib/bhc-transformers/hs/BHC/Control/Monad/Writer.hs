-- |
-- Module      : BHC.Control.Monad.Writer
-- Description : Writer monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Control.Monad.Writer (
    -- * The Writer monad
    Writer,
    runWriter,
    execWriter,
    mapWriter,
    
    -- * The WriterT monad transformer
    WriterT(..),
    runWriterT,
    execWriterT,
    mapWriterT,
    
    -- * Writer operations
    tell,
    listen,
    pass,
    listens,
    censor,
    writer,
    
    -- * MonadWriter class
    MonadWriter(..),
) where

import BHC.Prelude
import BHC.Control.Monad.Trans
import BHC.Control.Monad.Reader (Identity(..))

-- | The writer monad.
type Writer w = WriterT w Identity

-- | Run a Writer.
runWriter :: Writer w a -> (a, w)
runWriter = runIdentity . runWriterT

-- | Run and return only the output.
execWriter :: Writer w a -> w
execWriter = snd . runWriter

-- | Map both the return value and output.
mapWriter :: ((a, w) -> (b, w')) -> Writer w a -> Writer w' b
mapWriter f = mapWriterT (Identity . f . runIdentity)

-- | The writer monad transformer.
newtype WriterT w m a = WriterT { runWriterT :: m (a, w) }

instance (Functor m) => Functor (WriterT w m) where
    fmap f (WriterT m) = WriterT $ fmap (\(a, w) -> (f a, w)) m

instance (Monoid w, Applicative m) => Applicative (WriterT w m) where
    pure a = WriterT $ pure (a, mempty)
    WriterT mf <*> WriterT mx = WriterT $
        (\(f, w1) (x, w2) -> (f x, w1 <> w2)) <$> mf <*> mx

instance (Monoid w, Monad m) => Monad (WriterT w m) where
    WriterT m >>= k = WriterT $ do
        (a, w1) <- m
        (b, w2) <- runWriterT (k a)
        return (b, w1 <> w2)

instance Monoid w => MonadTrans (WriterT w) where
    lift m = WriterT $ do
        a <- m
        return (a, mempty)

instance (Monoid w, MonadIO m) => MonadIO (WriterT w m) where
    liftIO = lift . liftIO

-- | Run and return only the output.
execWriterT :: Monad m => WriterT w m a -> m w
execWriterT = fmap snd . runWriterT

-- | Map the inner computation.
mapWriterT :: (m (a, w) -> n (b, w')) -> WriterT w m a -> WriterT w' n b
mapWriterT f (WriterT m) = WriterT (f m)

-- | Append to the output.
tell :: (Monad m) => w -> WriterT w m ()
tell w = WriterT $ return ((), w)

-- | Execute and collect the output.
listen :: (Monad m) => WriterT w m a -> WriterT w m (a, w)
listen (WriterT m) = WriterT $ do
    (a, w) <- m
    return ((a, w), w)

-- | Execute with a function that can modify output.
pass :: (Monad m) => WriterT w m (a, w -> w) -> WriterT w m a
pass (WriterT m) = WriterT $ do
    ((a, f), w) <- m
    return (a, f w)

-- | Execute and apply a function to the output.
listens :: (Monad m) => (w -> b) -> WriterT w m a -> WriterT w m (a, b)
listens f = fmap (\(a, w) -> (a, f w)) . listen

-- | Apply a function to the output.
censor :: (Monad m) => (w -> w) -> WriterT w m a -> WriterT w m a
censor f m = pass $ fmap (\a -> (a, f)) m

-- | Create writer from a pair.
writer :: (Monad m) => (a, w) -> WriterT w m a
writer = WriterT . return

-- | Class for monads that can write.
class (Monoid w, Monad m) => MonadWriter w m | m -> w where
    tellW :: w -> m ()
    listenW :: m a -> m (a, w)
    passW :: m (a, w -> w) -> m a

instance (Monoid w, Monad m) => MonadWriter w (WriterT w m) where
    tellW = tell
    listenW = listen
    passW = pass
