-- |
-- Module      : Control.Monad.Writer
-- Description : Writer monads, overloaded
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.Writer
  ( MonadWriter (..)
  , listens
  , censor
  , Writer
  , runWriter
  , execWriter
  , WriterT (WriterT)
  , runWriterT
  , execWriterT
  , mapWriterT
  , module Control.Monad
  , module Control.Monad.Trans
  ) where

import Control.Monad
import Control.Monad.Trans
import Control.Monad.Trans.Writer.Strict
  ( Writer, WriterT (..), execWriter, execWriterT, mapWriterT, runWriter )
import Control.Monad.Writer.Class
