-- |
-- Module      : H26.Concurrency
-- Description : Structured concurrency primitives
-- License     : BSD-3-Clause
--
-- The H26.Concurrency module provides structured concurrency with
-- automatic cancellation propagation, deadline support, and safe
-- resource management. All concurrent operations happen within
-- scopes that outlive them.

{-# HASKELL_EDITION 2026 #-}
{-# PROFILE Server #-}

module H26.Concurrency
  ( -- * Scopes
    Scope
  , withScope
  , withScopeNamed
  , cancelScope
  , scopeName

    -- * Tasks
  , Task
  , spawn
  , spawnNamed
  , await
  , awaitAll
  , poll
  , cancel
  , taskId
  , taskStatus

    -- * Task Status
  , TaskStatus(..)
  , isRunning
  , isCompleted
  , isCancelled
  , isFailed

    -- * Cancellation
  , Cancellable(..)
  , checkCancelled
  , withCancellation
  , onCancel
  , CancelledException(..)
  , isCancelledException

    -- * Deadlines
  , withDeadline
  , withTimeout
  , Deadline
  , getDeadline
  , remainingTime
  , DeadlineExceeded(..)

    -- * Async Operations
  , async
  , asyncNamed
  , wait
  , waitCatch
  , waitAny
  , waitAll
  , waitEither
  , race
  , race_
  , concurrently
  , concurrently_
  , mapConcurrently
  , mapConcurrently_
  , forConcurrently
  , forConcurrently_
  , replicateConcurrently
  , replicateConcurrently_

    -- * Channels
  , Chan
  , newChan
  , readChan
  , writeChan
  , tryReadChan
  , tryWriteChan
  , isEmptyChan
  , dupChan

    -- * Bounded Channels
  , BoundedChan
  , newBoundedChan
  , readBoundedChan
  , writeBoundedChan
  , tryReadBoundedChan
  , tryWriteBoundedChan
  , isFull
  , capacity

    -- * MVars (Mutable Variables)
  , MVar
  , newMVar
  , newEmptyMVar
  , takeMVar
  , putMVar
  , readMVar
  , tryTakeMVar
  , tryPutMVar
  , tryReadMVar
  , isEmptyMVar
  , modifyMVar
  , modifyMVar_
  , withMVar
  , swapMVar

    -- * TVars (Transactional Variables)
  , TVar
  , newTVar
  , newTVarIO
  , readTVar
  , readTVarIO
  , writeTVar
  , modifyTVar
  , modifyTVar'

    -- * STM (Software Transactional Memory)
  , STM
  , atomically
  , retry
  , orElse
  , check
  , throwSTM
  , catchSTM

    -- * Semaphores
  , Semaphore
  , newSemaphore
  , acquireSemaphore
  , releaseSemaphore
  , tryAcquireSemaphore
  , withSemaphore

    -- * Mutexes
  , Mutex
  , newMutex
  , lock
  , unlock
  , tryLock
  , withMutex

    -- * Read-Write Locks
  , RWLock
  , newRWLock
  , readLock
  , readUnlock
  , writeLock
  , writeUnlock
  , withReadLock
  , withWriteLock

    -- * Condition Variables
  , Condition
  , newCondition
  , waitCondition
  , waitConditionFor
  , signal
  , broadcast

    -- * Barriers
  , Barrier
  , newBarrier
  , waitBarrier

    -- * Once (Single Initialization)
  , Once
  , newOnce
  , runOnce

    -- * Atomics
  , AtomicInt
  , newAtomicInt
  , readAtomicInt
  , writeAtomicInt
  , atomicAdd
  , atomicSub
  , atomicAnd
  , atomicOr
  , atomicXor
  , compareAndSwap

    -- * Atomic References
  , AtomicRef
  , newAtomicRef
  , readAtomicRef
  , writeAtomicRef
  , atomicModifyRef
  , atomicModifyRef'
  , casRef

    -- * Thread Utilities
  , ThreadId
  , myThreadId
  , forkIO
  , forkOS
  , forkOn
  , forkFinally
  , killThread
  , yield
  , threadDelay
  , threadCapability
  , setNumCapabilities
  , getNumCapabilities
  , getNumProcessors
  , isCurrentThreadBound
  , runInBoundThread
  , runInUnboundThread

    -- * Exception Handling
  , mask
  , mask_
  , uninterruptibleMask
  , uninterruptibleMask_
  , bracket
  , bracket_
  , bracketOnError
  , finally
  , onException

    -- * Parallel Evaluation
  , par
  , pseq
  , parMap
  , parList
  , parBuffer
  , parListChunk
  , NFData(..)
  , rseq
  , rpar
  , rdeepseq
  , using
  , withStrategy
  , runEval
  ) where

-- | Concurrency scope for structured concurrency.
--
-- All tasks spawned within a scope are guaranteed to complete
-- or be cancelled before the scope exits.
data Scope

-- | Run action within a new scope.
--
-- All spawned tasks complete before this returns.
-- If an exception occurs, all tasks are cancelled.
withScope :: (Scope -> IO a) -> IO a

-- | Run action within a named scope (for debugging).
withScopeNamed :: String -> (Scope -> IO a) -> IO a

-- | Cancel all tasks in scope.
cancelScope :: Scope -> IO ()

-- | Get scope name (if any).
scopeName :: Scope -> Maybe String

-- | Handle to a spawned task.
data Task a

-- | Spawn a task within a scope.
spawn :: Scope -> IO a -> IO (Task a)

-- | Spawn a named task.
spawnNamed :: String -> Scope -> IO a -> IO (Task a)

-- | Wait for task completion.
await :: Task a -> IO a

-- | Wait for all tasks.
awaitAll :: [Task a] -> IO [a]

-- | Check if task completed (non-blocking).
poll :: Task a -> IO (Maybe (Either SomeException a))

-- | Cancel a task.
cancel :: Task a -> IO ()

-- | Get task identifier.
taskId :: Task a -> TaskId

-- | Get current task status.
taskStatus :: Task a -> IO TaskStatus

-- | Task identifier.
newtype TaskId = TaskId Word64
  deriving (Eq, Ord, Show)

-- | Status of a task.
data TaskStatus
  = TaskPending     -- ^ Not yet started
  | TaskRunning     -- ^ Currently executing
  | TaskCompleted   -- ^ Finished successfully
  | TaskCancelled   -- ^ Was cancelled
  | TaskFailed      -- ^ Threw an exception
  deriving (Eq, Show)

-- | Check if task is running.
isRunning :: TaskStatus -> Bool

-- | Check if task completed successfully.
isCompleted :: TaskStatus -> Bool

-- | Check if task was cancelled.
isCancelled :: TaskStatus -> Bool

-- | Check if task failed with exception.
isFailed :: TaskStatus -> Bool

-- | Types that can be cancelled.
class Cancellable a where
  -- | Request cancellation.
  requestCancel :: a -> IO ()
  -- | Check if cancelled.
  wasCancelled :: a -> IO Bool

-- | Check if current task is cancelled.
--
-- Throws CancelledException if cancelled.
checkCancelled :: IO ()

-- | Run action with cancellation support.
withCancellation :: IO a -> IO a

-- | Register cleanup handler for cancellation.
onCancel :: IO a -> IO () -> IO a

-- | Exception thrown when task is cancelled.
data CancelledException = CancelledException
  deriving (Show, Eq)

instance Exception CancelledException

-- | Check if exception is cancellation.
isCancelledException :: SomeException -> Bool

-- | Run action with deadline.
--
-- Returns Nothing if deadline exceeded.
withDeadline :: Duration -> IO a -> IO (Maybe a)

-- | Run action with timeout.
--
-- Throws DeadlineExceeded if timeout exceeded.
withTimeout :: Duration -> IO a -> IO a

-- | Deadline timestamp.
data Deadline

-- | Get current deadline (if any).
getDeadline :: IO (Maybe Deadline)

-- | Time remaining until deadline.
remainingTime :: Deadline -> IO Duration

-- | Exception for deadline exceeded.
data DeadlineExceeded = DeadlineExceeded
  deriving (Show, Eq)

instance Exception DeadlineExceeded

-- | Spawn async computation.
async :: IO a -> IO (Task a)

-- | Spawn named async computation.
asyncNamed :: String -> IO a -> IO (Task a)

-- | Wait for async result.
wait :: Task a -> IO a

-- | Wait for async result, catching exceptions.
waitCatch :: Task a -> IO (Either SomeException a)

-- | Wait for any task to complete.
waitAny :: [Task a] -> IO (Task a, a)

-- | Wait for all tasks to complete.
waitAll :: [Task a] -> IO [a]

-- | Wait for either task to complete.
waitEither :: Task a -> Task b -> IO (Either a b)

-- | Race two actions, cancel loser.
race :: IO a -> IO b -> IO (Either a b)

-- | Race two actions, discard result.
race_ :: IO a -> IO b -> IO ()

-- | Run two actions concurrently.
concurrently :: IO a -> IO b -> IO (a, b)

-- | Run two actions concurrently, discard results.
concurrently_ :: IO a -> IO b -> IO ()

-- | Map function over list concurrently.
mapConcurrently :: (a -> IO b) -> [a] -> IO [b]

-- | Map function over list concurrently, discard results.
mapConcurrently_ :: (a -> IO b) -> [a] -> IO ()

-- | Flipped mapConcurrently.
forConcurrently :: [a] -> (a -> IO b) -> IO [b]

-- | Flipped mapConcurrently_.
forConcurrently_ :: [a] -> (a -> IO b) -> IO ()

-- | Replicate action concurrently.
replicateConcurrently :: Int -> IO a -> IO [a]

-- | Replicate action concurrently, discard results.
replicateConcurrently_ :: Int -> IO a -> IO ()

-- | Unbounded channel.
data Chan a

-- | Create new channel.
newChan :: IO (Chan a)

-- | Read from channel (blocks if empty).
readChan :: Chan a -> IO a

-- | Write to channel.
writeChan :: Chan a -> a -> IO ()

-- | Try to read (non-blocking).
tryReadChan :: Chan a -> IO (Maybe a)

-- | Try to write (non-blocking).
tryWriteChan :: Chan a -> a -> IO Bool

-- | Check if channel is empty.
isEmptyChan :: Chan a -> IO Bool

-- | Duplicate channel (new read end).
dupChan :: Chan a -> IO (Chan a)

-- | Bounded channel with capacity limit.
data BoundedChan a

-- | Create bounded channel.
newBoundedChan :: Int -> IO (BoundedChan a)

-- | Read from bounded channel.
readBoundedChan :: BoundedChan a -> IO a

-- | Write to bounded channel (blocks if full).
writeBoundedChan :: BoundedChan a -> a -> IO ()

-- | Try to read from bounded channel.
tryReadBoundedChan :: BoundedChan a -> IO (Maybe a)

-- | Try to write to bounded channel.
tryWriteBoundedChan :: BoundedChan a -> a -> IO Bool

-- | Check if channel is at capacity.
isFull :: BoundedChan a -> IO Bool

-- | Get channel capacity.
capacity :: BoundedChan a -> Int

-- | Mutable variable for synchronization.
data MVar a

-- | Create MVar with initial value.
newMVar :: a -> IO (MVar a)

-- | Create empty MVar.
newEmptyMVar :: IO (MVar a)

-- | Take value from MVar (blocks if empty).
takeMVar :: MVar a -> IO a

-- | Put value into MVar (blocks if full).
putMVar :: MVar a -> a -> IO ()

-- | Read MVar without taking.
readMVar :: MVar a -> IO a

-- | Try to take (non-blocking).
tryTakeMVar :: MVar a -> IO (Maybe a)

-- | Try to put (non-blocking).
tryPutMVar :: MVar a -> a -> IO Bool

-- | Try to read (non-blocking).
tryReadMVar :: MVar a -> IO (Maybe a)

-- | Check if MVar is empty.
isEmptyMVar :: MVar a -> IO Bool

-- | Modify MVar contents atomically.
modifyMVar :: MVar a -> (a -> IO (a, b)) -> IO b

-- | Modify MVar contents atomically (no result).
modifyMVar_ :: MVar a -> (a -> IO a) -> IO ()

-- | Execute action with MVar value.
withMVar :: MVar a -> (a -> IO b) -> IO b

-- | Swap MVar contents.
swapMVar :: MVar a -> a -> IO a

-- | Transactional variable.
data TVar a

-- | Create TVar in STM.
newTVar :: a -> STM (TVar a)

-- | Create TVar in IO.
newTVarIO :: a -> IO (TVar a)

-- | Read TVar in STM.
readTVar :: TVar a -> STM a

-- | Read TVar in IO.
readTVarIO :: TVar a -> IO a

-- | Write TVar in STM.
writeTVar :: TVar a -> a -> STM ()

-- | Modify TVar.
modifyTVar :: TVar a -> (a -> a) -> STM ()

-- | Modify TVar strictly.
modifyTVar' :: TVar a -> (a -> a) -> STM ()

-- | Software transactional memory monad.
data STM a

-- | Execute STM transaction atomically.
atomically :: STM a -> IO a

-- | Retry transaction (block until TVars change).
retry :: STM a

-- | Try first, then second if first retries.
orElse :: STM a -> STM a -> STM a

-- | Retry if condition is false.
check :: Bool -> STM ()

-- | Throw exception in STM.
throwSTM :: Exception e => e -> STM a

-- | Catch exception in STM.
catchSTM :: Exception e => STM a -> (e -> STM a) -> STM a

-- | Counting semaphore.
data Semaphore

-- | Create semaphore with initial count.
newSemaphore :: Int -> IO Semaphore

-- | Acquire permit (decrement, blocks if zero).
acquireSemaphore :: Semaphore -> IO ()

-- | Release permit (increment).
releaseSemaphore :: Semaphore -> IO ()

-- | Try to acquire (non-blocking).
tryAcquireSemaphore :: Semaphore -> IO Bool

-- | Execute with acquired semaphore.
withSemaphore :: Semaphore -> IO a -> IO a

-- | Mutual exclusion lock.
data Mutex

-- | Create new mutex.
newMutex :: IO Mutex

-- | Acquire mutex lock.
lock :: Mutex -> IO ()

-- | Release mutex lock.
unlock :: Mutex -> IO ()

-- | Try to acquire lock (non-blocking).
tryLock :: Mutex -> IO Bool

-- | Execute with mutex locked.
withMutex :: Mutex -> IO a -> IO a

-- | Read-write lock.
data RWLock

-- | Create new read-write lock.
newRWLock :: IO RWLock

-- | Acquire read lock.
readLock :: RWLock -> IO ()

-- | Release read lock.
readUnlock :: RWLock -> IO ()

-- | Acquire write lock.
writeLock :: RWLock -> IO ()

-- | Release write lock.
writeUnlock :: RWLock -> IO ()

-- | Execute with read lock.
withReadLock :: RWLock -> IO a -> IO a

-- | Execute with write lock.
withWriteLock :: RWLock -> IO a -> IO a

-- | Condition variable.
data Condition

-- | Create new condition variable.
newCondition :: IO Condition

-- | Wait on condition (must hold associated mutex).
waitCondition :: Condition -> Mutex -> IO ()

-- | Wait on condition with timeout.
waitConditionFor :: Condition -> Mutex -> Duration -> IO Bool

-- | Signal one waiting thread.
signal :: Condition -> IO ()

-- | Signal all waiting threads.
broadcast :: Condition -> IO ()

-- | Barrier for synchronizing threads.
data Barrier

-- | Create barrier for n threads.
newBarrier :: Int -> IO Barrier

-- | Wait at barrier.
waitBarrier :: Barrier -> IO ()

-- | Single initialization.
data Once

-- | Create new Once.
newOnce :: IO Once

-- | Run action at most once.
runOnce :: Once -> IO a -> IO a

-- | Atomic integer.
data AtomicInt

-- | Create atomic integer.
newAtomicInt :: Int -> IO AtomicInt

-- | Read atomic integer.
readAtomicInt :: AtomicInt -> IO Int

-- | Write atomic integer.
writeAtomicInt :: AtomicInt -> Int -> IO ()

-- | Atomic add, returns old value.
atomicAdd :: AtomicInt -> Int -> IO Int

-- | Atomic subtract, returns old value.
atomicSub :: AtomicInt -> Int -> IO Int

-- | Atomic bitwise and, returns old value.
atomicAnd :: AtomicInt -> Int -> IO Int

-- | Atomic bitwise or, returns old value.
atomicOr :: AtomicInt -> Int -> IO Int

-- | Atomic bitwise xor, returns old value.
atomicXor :: AtomicInt -> Int -> IO Int

-- | Compare and swap, returns success.
compareAndSwap :: AtomicInt -> Int -> Int -> IO Bool

-- | Atomic reference.
data AtomicRef a

-- | Create atomic reference.
newAtomicRef :: a -> IO (AtomicRef a)

-- | Read atomic reference.
readAtomicRef :: AtomicRef a -> IO a

-- | Write atomic reference.
writeAtomicRef :: AtomicRef a -> a -> IO ()

-- | Atomically modify reference.
atomicModifyRef :: AtomicRef a -> (a -> (a, b)) -> IO b

-- | Atomically modify reference strictly.
atomicModifyRef' :: AtomicRef a -> (a -> (a, b)) -> IO b

-- | Compare and swap reference.
casRef :: AtomicRef a -> a -> a -> IO Bool

-- | Thread identifier.
data ThreadId
  deriving (Eq, Ord, Show)

-- | Get current thread ID.
myThreadId :: IO ThreadId

-- | Fork new thread.
forkIO :: IO () -> IO ThreadId

-- | Fork bound thread.
forkOS :: IO () -> IO ThreadId

-- | Fork on specific capability.
forkOn :: Int -> IO () -> IO ThreadId

-- | Fork with cleanup handler.
forkFinally :: IO a -> (Either SomeException a -> IO ()) -> IO ThreadId

-- | Kill thread with async exception.
killThread :: ThreadId -> IO ()

-- | Yield to scheduler.
yield :: IO ()

-- | Delay for microseconds.
threadDelay :: Int -> IO ()

-- | Get thread's capability.
threadCapability :: ThreadId -> IO (Int, Bool)

-- | Set number of capabilities.
setNumCapabilities :: Int -> IO ()

-- | Get number of capabilities.
getNumCapabilities :: IO Int

-- | Get number of processors.
getNumProcessors :: IO Int

-- | Check if current thread is bound.
isCurrentThreadBound :: IO Bool

-- | Run in bound thread.
runInBoundThread :: IO a -> IO a

-- | Run in unbound thread.
runInUnboundThread :: IO a -> IO a

-- | Mask async exceptions.
mask :: ((forall a. IO a -> IO a) -> IO b) -> IO b

-- | Mask async exceptions (no restore).
mask_ :: IO a -> IO a

-- | Mask async exceptions uninterruptibly.
uninterruptibleMask :: ((forall a. IO a -> IO a) -> IO b) -> IO b

-- | Mask async exceptions uninterruptibly (no restore).
uninterruptibleMask_ :: IO a -> IO a

-- | Acquire/release pattern.
bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c

-- | Bracket without resource.
bracket_ :: IO a -> IO b -> IO c -> IO c

-- | Bracket with cleanup only on error.
bracketOnError :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c

-- | Run action with cleanup.
finally :: IO a -> IO b -> IO a

-- | Run cleanup on exception.
onException :: IO a -> IO b -> IO a

-- | Spark parallel evaluation.
par :: a -> b -> b

-- | Parallel sequence.
pseq :: a -> b -> b

-- | Parallel map.
parMap :: (a -> b) -> [a] -> [b]

-- | Parallel list evaluation.
parList :: Strategy a -> Strategy [a]

-- | Parallel buffer.
parBuffer :: Int -> Strategy a -> Strategy [a]

-- | Parallel chunked list.
parListChunk :: Int -> Strategy a -> Strategy [a]

-- | Class for deep evaluation.
class NFData a where
  rnf :: a -> ()

-- | Evaluate to WHNF strategy.
rseq :: Strategy a

-- | Spark parallel evaluation strategy.
rpar :: Strategy a

-- | Evaluate to NF strategy.
rdeepseq :: NFData a => Strategy a

-- | Apply strategy to value.
using :: a -> Strategy a -> a

-- | Apply strategy in context.
withStrategy :: Strategy a -> a -> a

-- | Run evaluation.
runEval :: Eval a -> a

-- Internal types
type Strategy a = a -> Eval a
data Eval a
data Duration
data SomeException
class Exception e

-- This is a specification file.
-- Actual implementation provided by the compiler.
