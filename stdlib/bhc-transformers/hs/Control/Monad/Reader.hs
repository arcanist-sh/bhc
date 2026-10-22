-- |
-- Module      : Control.Monad.Reader
-- Description : Reader monads, overloaded
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.Reader
  ( MonadReader (..)
  , asks
  , Reader
  , runReader
  , mapReader
  , withReader
  , ReaderT (ReaderT)
  , runReaderT
  , mapReaderT
  , withReaderT
  , module Control.Monad
  , module Control.Monad.Trans
  ) where

import Control.Monad
import Control.Monad.Reader.Class
import Control.Monad.Trans
import Control.Monad.Trans.Reader
  ( Reader, ReaderT (..), mapReader, mapReaderT, runReader, withReader, withReaderT )
