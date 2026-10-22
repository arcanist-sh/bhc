-- |
-- Module      : Control.Monad.RWS.Strict
-- Description : Reader-writer-state monads, overloaded
-- License     : BSD-3-Clause
--
-- Part of BHC's mtl library (spec/BHC-BRIEF-0005).
module Control.Monad.RWS.Strict
  ( RWS
  , rws
  , runRWS
  , evalRWS
  , execRWS
  , RWST (RWST)
  , runRWST
  , evalRWST
  , execRWST
  , module Control.Monad.Reader.Class
  , module Control.Monad.State.Class
  , module Control.Monad.Writer.Class
  , module Control.Monad
  , module Control.Monad.Trans
  ) where

import Control.Monad
import Control.Monad.Reader.Class
import Control.Monad.State.Class
import Control.Monad.Trans
import Control.Monad.Trans.RWS.Strict
  ( RWS, RWST (..), evalRWS, evalRWST, execRWS, execRWST, runRWS, rws )
import Control.Monad.Writer.Class
