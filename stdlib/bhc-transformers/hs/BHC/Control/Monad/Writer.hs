-- |
-- Module      : BHC.Control.Monad.Writer
-- Description : Writer monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The Writer monad for computations that produce output.
--
-- = Overview
--
-- The Writer monad represents computations that accumulate output
-- (such as logs, traces, or audit trails) alongside their return value.
-- The output type must be a 'Monoid' so outputs can be combined.
--
-- = Usage
--
-- @
-- import BHC.Control.Monad.Writer
--
-- type Log = [String]
--
-- -- Computation that logs its progress
-- factorial :: Int -> Writer Log Int
-- factorial 0 = do
--     tell [\"Base case: 0! = 1\"]
--     return 1
-- factorial n = do
--     tell [\"Computing \" ++ show n ++ \"!\"]
--     rest <- factorial (n - 1)
--     return (n * rest)
--
-- -- Run and get both result and log
-- (result, log) = runWriter (factorial 5)
-- @
--
-- = Common Output Types
--
-- * @[a]@ — Collect values into a list
-- * @Sum Int@ — Accumulate a count
-- * @Endo a@ — Build a function by composition
-- * @DList a@ — Efficient list building

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
import BHC.Control.Monad.Identity (Identity(..))

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

-- | /O(1)/. Produce output without a result value.
--
-- The output is combined with any previous output using @(\<\>)@.
--
-- ==== __Examples__
--
-- >>> runWriter $ tell "hello" >> tell " world"
-- ((), "hello world")
--
-- @
-- logMessage :: String -> Writer [String] ()
-- logMessage msg = tell [msg]
-- @
tell :: (Monad m) => w -> WriterT w m ()
tell w = WriterT $ return ((), w)

-- | /O(1)/. Execute a computation and collect its output.
--
-- Returns both the computation's result and its output as a pair,
-- while also including the output in the overall writer state.
--
-- ==== __Examples__
--
-- >>> runWriter $ listen (tell "hi" >> return 42)
-- ((42, "hi"), "hi")
listen :: (Monad m) => WriterT w m a -> WriterT w m (a, w)
listen (WriterT m) = WriterT $ do
    (a, w) <- m
    return ((a, w), w)

-- | /O(1)/. Execute a computation that can transform its output.
--
-- The computation returns a pair @(result, transform)@. The transform
-- function is applied to the output before it is committed.
--
-- ==== __Examples__
--
-- >>> runWriter $ pass (return (42, map toUpper))
-- (42, "")
--
-- @
-- -- Reverse only this computation's output
-- reversed :: Writer String a -> Writer String a
-- reversed m = pass $ do
--     a <- m
--     return (a, reverse)
-- @
pass :: (Monad m) => WriterT w m (a, w -> w) -> WriterT w m a
pass (WriterT m) = WriterT $ do
    ((a, f), w) <- m
    return (a, f w)

-- | /O(1)/. Execute and apply a function to the output.
--
-- Combines 'listen' with @fmap@.
--
-- ==== __Examples__
--
-- >>> runWriter $ listens length (tell "hello")
-- (((), 5), "hello")
listens :: (Monad m) => (w -> b) -> WriterT w m a -> WriterT w m (a, b)
listens f = fmap (\(a, w) -> (a, f w)) . listen

-- | /O(1)/. Transform the output of a computation.
--
-- ==== __Examples__
--
-- >>> runWriter $ censor (map toUpper) (tell "hello")
-- ((), "HELLO")
censor :: (Monad m) => (w -> w) -> WriterT w m a -> WriterT w m a
censor f m = pass $ fmap (\a -> (a, f)) m

-- | /O(1)/. Create a writer from a result and output pair.
--
-- ==== __Examples__
--
-- >>> runWriter $ writer (42, "answer")
-- (42, "answer")
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
