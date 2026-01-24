-- |
-- Module      : BHC.Control.Monad.Reader
-- Description : Reader monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The Reader monad for computations that read from a shared environment.
--
-- = Overview
--
-- The Reader monad represents computations that depend on some shared
-- read-only environment. This is useful for:
--
-- * Configuration that needs to be available throughout a computation
-- * Database connections or handles that should be passed implicitly
-- * Context information like user credentials or request metadata
--
-- = Usage
--
-- @
-- import BHC.Control.Monad.Reader
--
-- data Config = Config { dbHost :: String, dbPort :: Int }
--
-- -- Computations that need configuration
-- getConnectionString :: Reader Config String
-- getConnectionString = do
--     host <- asks dbHost
--     port <- asks dbPort
--     return $ host ++ ":" ++ show port
--
-- -- Run with a specific config
-- main = print $ runReader getConnectionString config
-- @
--
-- = Transformer Usage
--
-- @
-- type App = ReaderT Config IO
--
-- runApp :: App a -> Config -> IO a
-- runApp = runReaderT
--
-- fetchData :: App Data
-- fetchData = do
--     connStr <- asks dbHost
--     liftIO $ queryDatabase connStr
-- @

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
--
-- @Reader r a@ is a computation that reads a value of type @r@ and
-- produces a value of type @a@.
type Reader r = ReaderT r Identity

-- | /O(1)/. Run a 'Reader' computation with the given environment.
--
-- ==== __Examples__
--
-- >>> runReader (asks length) "hello"
-- 5
runReader :: Reader r a -> r -> a
runReader m = runIdentity . runReaderT m

-- | /O(1)/. Map the return value of a 'Reader'.
--
-- ==== __Examples__
--
-- >>> runReader (mapReader (*2) (asks length)) "hello"
-- 10
mapReader :: (a -> b) -> Reader r a -> Reader r b
mapReader f = mapReaderT (Identity . f . runIdentity)

-- | /O(1)/. Execute a 'Reader' with a modified environment.
--
-- ==== __Examples__
--
-- >>> runReader (withReader reverse (asks head)) "hello"
-- 'o'
withReader :: (r' -> r) -> Reader r a -> Reader r' a
withReader = withReaderT

-- | The reader monad transformer.
--
-- Adds a read-only environment to an underlying monad @m@.
-- All operations in the underlying monad are available via 'lift'.
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

-- | /O(1)/. Retrieve the entire environment.
--
-- ==== __Examples__
--
-- >>> runReader ask "hello"
-- "hello"
--
-- @
-- greet :: Reader String String
-- greet = do
--     name <- ask
--     return $ "Hello, " ++ name
-- @
ask :: Monad m => ReaderT r m r
ask = ReaderT return

-- | /O(1)/. Execute a computation with a modified environment.
--
-- The modification only affects the given computation; subsequent
-- operations see the original environment.
--
-- ==== __Examples__
--
-- >>> runReader (local reverse ask) "hello"
-- "olleh"
--
-- @
-- withUpperCase :: Reader String a -> Reader String a
-- withUpperCase = local (map toUpper)
-- @
local :: (r -> r) -> ReaderT r m a -> ReaderT r m a
local f (ReaderT g) = ReaderT (g . f)

-- | /O(1)/. Retrieve a function of the environment.
--
-- @asks f@ is equivalent to @fmap f ask@, but more efficient.
--
-- ==== __Examples__
--
-- >>> runReader (asks length) "hello"
-- 5
--
-- @
-- getPort :: Reader Config Int
-- getPort = asks configPort
-- @
asks :: Monad m => (r -> a) -> ReaderT r m a
asks f = ReaderT (return . f)

-- | /O(1)/. Embed a simple reader action into the monad.
--
-- Synonym for 'asks'.
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
