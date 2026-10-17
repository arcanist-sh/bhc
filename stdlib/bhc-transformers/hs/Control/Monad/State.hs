-- |
-- Module      : Control.Monad.State
-- Description : State monads, overloaded
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.State
  ( MonadState (..)
  , modify
  , modify'
  , gets
  , State
  , runState
  , evalState
  , execState
  , mapState
  , withState
  , StateT (StateT)
  , runStateT
  , evalStateT
  , execStateT
  , mapStateT
  , withStateT
  , module Control.Monad
  , module Control.Monad.Trans
  ) where

import Control.Monad
import Control.Monad.State.Class
import Control.Monad.Trans
import Control.Monad.Trans.State.Strict
  ( State, StateT (..), evalState, evalStateT, execState, execStateT
  , mapState, mapStateT, runState, withState, withStateT )
