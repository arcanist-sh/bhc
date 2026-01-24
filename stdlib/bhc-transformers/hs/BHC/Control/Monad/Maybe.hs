-- |
-- Module      : BHC.Control.Monad.Maybe
-- Description : Maybe monad transformer
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The MaybeT monad transformer adds failure capability to a monad.
--
-- = Usage
--
-- @
-- import BHC.Control.Monad.Maybe
--
-- -- Look up a user, then their address
-- getUserAddress :: UserId -> MaybeT IO Address
-- getUserAddress uid = do
--     user <- MaybeT (lookupUser uid)       -- Returns Maybe User
--     MaybeT (lookupAddress (userId user))  -- Returns Maybe Address
-- @

module BHC.Control.Monad.Maybe
    ( -- * The MaybeT monad transformer
      MaybeT(..)
    , runMaybeT
    , mapMaybeT

      -- * MaybeT operations
    , maybeToExceptT
    , exceptToMaybeT
    , hoistMaybe

      -- * Lifting from Maybe
    , liftMaybe
    ) where

import BHC.Prelude
import BHC.Control.Monad.Trans
import BHC.Control.Monad.Except (ExceptT(..))

-- ============================================================
-- MaybeT monad transformer
-- ============================================================

-- | The Maybe monad transformer.
--
-- Computation type: Computations that may fail.
-- Binding strategy: Failures short-circuit the computation.
-- Useful for: Computations that may fail at any point.
newtype MaybeT m a = MaybeT { runMaybeT :: m (Maybe a) }

instance Functor m => Functor (MaybeT m) where
    fmap f (MaybeT m) = MaybeT (fmap (fmap f) m)
    {-# INLINE fmap #-}

instance Monad m => Applicative (MaybeT m) where
    pure = MaybeT . return . Just
    {-# INLINE pure #-}

    MaybeT mf <*> MaybeT mx = MaybeT $ do
        maybeF <- mf
        case maybeF of
            Nothing -> return Nothing
            Just f  -> fmap (fmap f) mx
    {-# INLINE (<*>) #-}

instance Monad m => Monad (MaybeT m) where
    MaybeT m >>= k = MaybeT $ do
        maybeA <- m
        case maybeA of
            Nothing -> return Nothing
            Just a  -> runMaybeT (k a)
    {-# INLINE (>>=) #-}

instance MonadTrans MaybeT where
    lift = MaybeT . fmap Just
    {-# INLINE lift #-}

instance MonadIO m => MonadIO (MaybeT m) where
    liftIO = lift . liftIO
    {-# INLINE liftIO #-}

-- | MonadPlus instance for MaybeT.
-- 'mzero' is the failing computation, 'mplus' tries the second if the first fails.
instance Monad m => Alternative (MaybeT m) where
    empty = MaybeT (return Nothing)
    {-# INLINE empty #-}

    MaybeT m1 <|> MaybeT m2 = MaybeT $ do
        maybeA <- m1
        case maybeA of
            Nothing -> m2
            Just a  -> return (Just a)
    {-# INLINE (<|>) #-}

instance Monad m => MonadPlus (MaybeT m) where
    mzero = empty
    {-# INLINE mzero #-}

    mplus = (<|>)
    {-# INLINE mplus #-}

instance Foldable m => Foldable (MaybeT m) where
    foldMap f (MaybeT m) = foldMap (foldMap f) m
    {-# INLINE foldMap #-}

instance Traversable m => Traversable (MaybeT m) where
    traverse f (MaybeT m) = MaybeT <$> traverse (traverse f) m
    {-# INLINE traverse #-}

-- ============================================================
-- MaybeT operations
-- ============================================================

-- | Transform the computation inside a MaybeT.
--
-- ==== __Examples__
--
-- >>> mapMaybeT (fmap (fmap (+1))) (MaybeT [Just 1, Nothing, Just 2])
-- MaybeT [Just 2, Nothing, Just 3]
mapMaybeT :: (m (Maybe a) -> n (Maybe b)) -> MaybeT m a -> MaybeT n b
mapMaybeT f (MaybeT m) = MaybeT (f m)
{-# INLINE mapMaybeT #-}

-- | Convert a 'MaybeT' to an 'ExceptT'.
--
-- The error value is provided for the 'Nothing' case.
--
-- ==== __Examples__
--
-- >>> runExceptT (maybeToExceptT "error" (MaybeT (Just (Just 1))))
-- Identity (Right 1)
--
-- >>> runExceptT (maybeToExceptT "error" (MaybeT (Just Nothing)))
-- Identity (Left "error")
maybeToExceptT :: Functor m => e -> MaybeT m a -> ExceptT e m a
maybeToExceptT e (MaybeT m) = ExceptT (fmap (maybe (Left e) Right) m)
{-# INLINE maybeToExceptT #-}

-- | Convert an 'ExceptT' to a 'MaybeT', discarding the error.
--
-- ==== __Examples__
--
-- >>> runMaybeT (exceptToMaybeT (ExceptT (Just (Right 1))))
-- Identity (Just 1)
--
-- >>> runMaybeT (exceptToMaybeT (ExceptT (Just (Left "error"))))
-- Identity Nothing
exceptToMaybeT :: Functor m => ExceptT e m a -> MaybeT m a
exceptToMaybeT (ExceptT m) = MaybeT (fmap (either (const Nothing) Just) m)
{-# INLINE exceptToMaybeT #-}

-- | Lift a 'Maybe' into a 'MaybeT'.
--
-- This is equivalent to @MaybeT . return@.
--
-- ==== __Examples__
--
-- >>> runMaybeT (hoistMaybe (Just 1)) :: IO (Maybe Int)
-- Just 1
--
-- >>> runMaybeT (hoistMaybe Nothing) :: IO (Maybe Int)
-- Nothing
hoistMaybe :: Applicative m => Maybe a -> MaybeT m a
hoistMaybe = MaybeT . pure
{-# INLINE hoistMaybe #-}

-- | Lift a 'Maybe' into any 'MonadPlus'.
--
-- ==== __Examples__
--
-- >>> liftMaybe (Just 1) :: [Int]
-- [1]
--
-- >>> liftMaybe Nothing :: [Int]
-- []
liftMaybe :: MonadPlus m => Maybe a -> m a
liftMaybe Nothing  = mzero
liftMaybe (Just a) = return a
{-# INLINE liftMaybe #-}
