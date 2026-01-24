-- |
-- Module      : BHC.Control.Monad.Trans
-- Description : Monad transformers
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Core type classes for monad transformers.
--
-- = Overview
--
-- Monad transformers allow you to combine multiple monadic effects
-- into a single monad. For example, @StateT s (ReaderT r IO)@
-- combines state, a read-only environment, and IO.
--
-- = Lifting Operations
--
-- When working with transformer stacks, use 'lift' to access
-- operations from the underlying monad:
--
-- @
-- type App = StateT Int (ReaderT Config IO)
--
-- example :: App ()
-- example = do
--     n <- get                        -- StateT operation
--     cfg <- lift ask                 -- ReaderT operation
--     lift $ lift $ putStrLn "hello"  -- IO operation
--     liftIO $ putStrLn "hello"       -- Or use liftIO directly
-- @
--
-- = MonadIO
--
-- 'MonadIO' provides a single 'liftIO' that lifts through any
-- number of transformer layers, avoiding nested 'lift' calls.

module BHC.Control.Monad.Trans (
    -- * MonadTrans
    MonadTrans(..),

    -- * MonadIO
    MonadIO(..),
) where

import BHC.Prelude

-- | Class of monad transformers.
--
-- Monad transformers take a monad @m@ and produce a new monad @t m@
-- with additional capabilities.
--
-- ==== __Laws__
--
-- @
-- lift . return  ≡  return
-- lift (m >>= f) ≡  lift m >>= (lift . f)
-- @
class MonadTrans t where
    -- | /O(1)/. Lift a computation from the inner monad.
    --
    -- ==== __Examples__
    --
    -- >>> evalStateT (lift [1,2,3]) 0
    -- [1, 2, 3]
    lift :: Monad m => m a -> t m a

-- | Class of monads that can perform IO.
--
-- This class allows lifting IO actions through any number of
-- transformer layers with a single 'liftIO' call.
--
-- ==== __Laws__
--
-- @
-- liftIO . return  ≡  return
-- liftIO (m >>= f) ≡  liftIO m >>= (liftIO . f)
-- @
class Monad m => MonadIO m where
    -- | /O(1)/. Lift an IO action into the monad.
    --
    -- ==== __Examples__
    --
    -- @
    -- greet :: StateT Int IO ()
    -- greet = do
    --     n <- get
    --     liftIO $ putStrLn ("Count: " ++ show n)
    -- @
    liftIO :: IO a -> m a

instance MonadIO IO where
    liftIO = id
