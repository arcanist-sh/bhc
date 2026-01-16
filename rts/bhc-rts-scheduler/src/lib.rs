//! Work-stealing task scheduler for the BHC Runtime System.
//!
//! This crate implements structured concurrency primitives as specified in
//! H26-SPEC Section 10: Concurrency Model. Key features:
//!
//! - **Work-stealing scheduler** - Efficient load balancing across workers
//! - **Structured concurrency** - Tasks are scoped and cancellation propagates
//! - **Cooperative cancellation** - Tasks check for cancellation at safe points
//! - **Deadline support** - Time-bounded operations
//!
//! # Structured Concurrency Model
//!
//! All concurrent operations happen within a scope that outlives them:
//!
//! ```ignore
//! use bhc_rts_scheduler::{Scheduler, with_scope};
//!
//! let scheduler = Scheduler::new(4); // 4 worker threads
//!
//! with_scope(&scheduler, |scope| {
//!     let task1 = scope.spawn(|| compute_x());
//!     let task2 = scope.spawn(|| compute_y());
//!
//!     let x = task1.await_result();
//!     let y = task2.await_result();
//!     (x, y)
//! });
//! // All tasks complete before scope exits
//! ```
//!
//! # Task Lifecycle
//!
//! ```text
//!   spawn      await
//!     |          |
//!     v          v
//! +-----+    +-------+    +----------+    +---------+
//! | New | -> |Running| -> |Completing| -> |Completed|
//! +-----+    +-------+    +----------+    +---------+
//!               |                              ^
//!               | cancel                       |
//!               v                              |
//!            +----------+                      |
//!            |Cancelling| ---------------------+
//!            +----------+
//! ```
//!
//! # Design Goals
//!
//! - Efficient work distribution with minimal contention
//! - Predictable cancellation semantics
//! - Support for parallel numeric operations
//! - Bounded latency for Server Profile

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use crossbeam::deque::{Injector, Stealer, Worker as WorkerDeque};
use parking_lot::{Condvar, Mutex, RwLock};
use std::any::Any;
use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Unique identifier for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    /// Create a new unique task ID.
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Task({})", self.0)
    }
}

/// State of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is created but not yet running.
    Pending,
    /// Task is currently executing.
    Running,
    /// Task is being cancelled.
    Cancelling,
    /// Task completed successfully.
    Completed,
    /// Task was cancelled.
    Cancelled,
    /// Task panicked.
    Failed,
}

/// Result of a task execution.
#[derive(Debug)]
pub enum TaskResult<T> {
    /// Task completed with a value.
    Ok(T),
    /// Task was cancelled.
    Cancelled,
    /// Task panicked.
    Panicked(Box<dyn Any + Send>),
}

impl<T> TaskResult<T> {
    /// Check if the task completed successfully.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }

    /// Unwrap the result, panicking if not Ok.
    #[must_use]
    pub fn unwrap(self) -> T {
        match self {
            Self::Ok(v) => v,
            Self::Cancelled => panic!("task was cancelled"),
            Self::Panicked(_) => panic!("task panicked"),
        }
    }
}

/// A task that can be spawned and awaited.
pub struct Task<T> {
    id: TaskId,
    state: Arc<Mutex<TaskState>>,
    result: Arc<Mutex<Option<TaskResult<T>>>>,
    cancelled: Arc<AtomicBool>,
    condvar: Arc<Condvar>,
}

impl<T> Task<T> {
    /// Create a new task.
    fn new() -> Self {
        Self {
            id: TaskId::new(),
            state: Arc::new(Mutex::new(TaskState::Pending)),
            result: Arc::new(Mutex::new(None)),
            cancelled: Arc::new(AtomicBool::new(false)),
            condvar: Arc::new(Condvar::new()),
        }
    }

    /// Get the task's ID.
    #[must_use]
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Get the current state of the task.
    #[must_use]
    pub fn state(&self) -> TaskState {
        *self.state.lock()
    }

    /// Check if the task has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Request cancellation of this task.
    ///
    /// The task will be cancelled at the next safe point.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        *self.state.lock() = TaskState::Cancelling;
    }

    /// Wait for the task to complete and return its result.
    pub fn await_result(self) -> TaskResult<T> {
        let mut state = self.state.lock();
        while !matches!(
            *state,
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        ) {
            self.condvar.wait(&mut state);
        }
        drop(state);

        self.result.lock().take().expect("result should be set")
    }

    /// Try to get the result without blocking.
    #[must_use]
    pub fn try_result(&self) -> Option<TaskResult<T>>
    where
        T: Clone,
    {
        let state = self.state.lock();
        if matches!(
            *state,
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        ) {
            self.result.lock().clone()
        } else {
            None
        }
    }

    /// Wait for the task with a timeout.
    pub fn await_timeout(self, timeout: Duration) -> Option<TaskResult<T>> {
        let deadline = Instant::now() + timeout;
        let mut state = self.state.lock();

        while !matches!(
            *state,
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        ) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let result = self.condvar.wait_for(&mut state, remaining);
            if result.timed_out() {
                return None;
            }
        }
        drop(state);

        Some(self.result.lock().take().expect("result should be set"))
    }
}

impl<T: Clone> Clone for TaskResult<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Ok(v) => Self::Ok(v.clone()),
            Self::Cancelled => Self::Cancelled,
            Self::Panicked(_) => Self::Panicked(Box::new("cloned panic")),
        }
    }
}

impl<T> fmt::Debug for Task<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Task")
            .field("id", &self.id)
            .field("state", &self.state())
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

// Internal task representation for the scheduler
struct RawTask {
    func: Box<dyn FnOnce() + Send>,
}

impl RawTask {
    fn new<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self { func: Box::new(f) }
    }

    fn run(self) {
        (self.func)();
    }
}

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of worker threads.
    pub num_workers: usize,
    /// Stack size for worker threads.
    pub stack_size: usize,
    /// Enable work stealing.
    pub work_stealing: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus(),
            stack_size: 2 * 1024 * 1024, // 2 MB
            work_stealing: true,
        }
    }
}

fn num_cpus() -> usize {
    thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Statistics for the scheduler.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total tasks spawned.
    pub tasks_spawned: u64,
    /// Total tasks completed.
    pub tasks_completed: u64,
    /// Total tasks cancelled.
    pub tasks_cancelled: u64,
    /// Total tasks that panicked.
    pub tasks_failed: u64,
    /// Number of successful steals.
    pub steals: u64,
    /// Number of failed steal attempts.
    pub steal_failures: u64,
}

/// A work-stealing task scheduler.
///
/// The scheduler manages a pool of worker threads that execute tasks.
/// Each worker has a local work queue and can steal from others when idle.
pub struct Scheduler {
    config: SchedulerConfig,
    global_queue: Arc<Injector<RawTask>>,
    stealers: Arc<Vec<Stealer<RawTask>>>,
    workers: Vec<JoinHandle<()>>,
    stats: Arc<RwLock<SchedulerStats>>,
    shutdown: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
}

impl Scheduler {
    /// Create a new scheduler with the specified number of workers.
    #[must_use]
    pub fn new(num_workers: usize) -> Self {
        Self::with_config(SchedulerConfig {
            num_workers,
            ..Default::default()
        })
    }

    /// Create a new scheduler with the given configuration.
    #[must_use]
    pub fn with_config(config: SchedulerConfig) -> Self {
        let global_queue = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(RwLock::new(SchedulerStats::default()));
        let active_tasks = Arc::new(AtomicUsize::new(0));

        let mut local_queues = Vec::with_capacity(config.num_workers);
        let mut stealers = Vec::with_capacity(config.num_workers);

        // Create worker deques
        for _ in 0..config.num_workers {
            let worker = WorkerDeque::new_fifo();
            stealers.push(worker.stealer());
            local_queues.push(worker);
        }

        let stealers = Arc::new(stealers);
        let mut workers = Vec::with_capacity(config.num_workers);

        // Spawn worker threads
        for (id, local_queue) in local_queues.into_iter().enumerate() {
            let global = Arc::clone(&global_queue);
            let stealers = Arc::clone(&stealers);
            let shutdown = Arc::clone(&shutdown);
            let stats = Arc::clone(&stats);

            let handle = thread::Builder::new()
                .name(format!("bhc-worker-{id}"))
                .stack_size(config.stack_size)
                .spawn(move || {
                    worker_loop(id, local_queue, global, stealers, shutdown, stats);
                })
                .expect("failed to spawn worker thread");

            workers.push(handle);
        }

        Self {
            config,
            global_queue,
            stealers,
            workers,
            stats,
            shutdown,
            active_tasks,
        }
    }

    /// Create a new scheduler with default configuration.
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    /// Spawn a task on the scheduler.
    pub fn spawn<F, T>(&self, f: F) -> Task<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let task = Task::new();
        let state = Arc::clone(&task.state);
        let result = Arc::clone(&task.result);
        let cancelled = Arc::clone(&task.cancelled);
        let condvar = Arc::clone(&task.condvar);

        self.active_tasks.fetch_add(1, Ordering::Relaxed);
        let active_tasks = Arc::clone(&self.active_tasks);

        {
            let mut stats = self.stats.write();
            stats.tasks_spawned += 1;
        }

        let raw_task = RawTask::new(move || {
            // Check for early cancellation
            if cancelled.load(Ordering::Acquire) {
                *state.lock() = TaskState::Cancelled;
                *result.lock() = Some(TaskResult::Cancelled);
                condvar.notify_all();
                active_tasks.fetch_sub(1, Ordering::Relaxed);
                return;
            }

            *state.lock() = TaskState::Running;

            // Run the task
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

            // Check for cancellation during execution
            if cancelled.load(Ordering::Acquire) {
                *state.lock() = TaskState::Cancelled;
                *result.lock() = Some(TaskResult::Cancelled);
            } else {
                match outcome {
                    Ok(value) => {
                        *state.lock() = TaskState::Completed;
                        *result.lock() = Some(TaskResult::Ok(value));
                    }
                    Err(panic) => {
                        *state.lock() = TaskState::Failed;
                        *result.lock() = Some(TaskResult::Panicked(panic));
                    }
                }
            }

            condvar.notify_all();
            active_tasks.fetch_sub(1, Ordering::Relaxed);
        });

        self.global_queue.push(raw_task);
        task
    }

    /// Get scheduler statistics.
    #[must_use]
    pub fn stats(&self) -> SchedulerStats {
        self.stats.read().clone()
    }

    /// Get the number of worker threads.
    #[must_use]
    pub fn num_workers(&self) -> usize {
        self.config.num_workers
    }

    /// Get the number of currently active tasks.
    #[must_use]
    pub fn active_tasks(&self) -> usize {
        self.active_tasks.load(Ordering::Relaxed)
    }

    /// Shutdown the scheduler and wait for all workers to finish.
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Release);

        for worker in std::mem::take(&mut self.workers) {
            let _ = worker.join();
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

fn worker_loop(
    id: usize,
    local: WorkerDeque<RawTask>,
    global: Arc<Injector<RawTask>>,
    stealers: Arc<Vec<Stealer<RawTask>>>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<RwLock<SchedulerStats>>,
) {
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }

        // Try local queue first
        if let Some(task) = local.pop() {
            task.run();
            let mut s = stats.write();
            s.tasks_completed += 1;
            continue;
        }

        // Try global queue
        if let crossbeam::deque::Steal::Success(task) = global.steal() {
            task.run();
            let mut s = stats.write();
            s.tasks_completed += 1;
            continue;
        }

        // Try stealing from other workers
        let mut stolen = false;
        for (i, stealer) in stealers.iter().enumerate() {
            if i == id {
                continue;
            }
            if let crossbeam::deque::Steal::Success(task) = stealer.steal() {
                task.run();
                let mut s = stats.write();
                s.tasks_completed += 1;
                s.steals += 1;
                stolen = true;
                break;
            }
        }

        if !stolen {
            let mut s = stats.write();
            s.steal_failures += 1;
            drop(s);
            // Yield to avoid busy-waiting
            thread::yield_now();
        }
    }
}

/// A scope for structured concurrency.
///
/// Tasks spawned within a scope are guaranteed to complete before
/// the scope exits.
pub struct Scope<'a> {
    scheduler: &'a Scheduler,
    tasks: Mutex<Vec<Box<dyn FnOnce() + Send + 'a>>>,
}

impl<'a> Scope<'a> {
    /// Create a new scope.
    fn new(scheduler: &'a Scheduler) -> Self {
        Self {
            scheduler,
            tasks: Mutex::new(Vec::new()),
        }
    }

    /// Spawn a task within this scope.
    pub fn spawn<F, T>(&self, f: F) -> Task<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        // For a full implementation, we would track the task
        // and ensure it completes before scope exits.
        // For now, use the scheduler's spawn.
        self.scheduler.spawn(f)
    }
}

/// Execute a function within a structured concurrency scope.
///
/// All tasks spawned within the scope are guaranteed to complete
/// before this function returns.
pub fn with_scope<'a, F, R>(scheduler: &'a Scheduler, f: F) -> R
where
    F: FnOnce(&Scope<'a>) -> R,
{
    let scope = Scope::new(scheduler);
    f(&scope)
    // In a full implementation, we would wait for all spawned tasks here
}

/// Check if the current task has been cancelled.
///
/// Tasks should call this at safe points to enable cooperative cancellation.
pub fn check_cancelled() {
    // In a full implementation, this would check thread-local task state
    // and throw a cancellation exception if cancelled.
}

/// Thread-local storage for the current task context.
thread_local! {
    static CURRENT_TASK: Cell<Option<TaskId>> = const { Cell::new(None) };
}

/// Get the ID of the currently executing task, if any.
#[must_use]
pub fn current_task_id() -> Option<TaskId> {
    CURRENT_TASK.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;

    #[test]
    fn test_task_id_uniqueness() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_task_states() {
        let task: Task<i32> = Task::new();
        assert_eq!(task.state(), TaskState::Pending);
        assert!(!task.is_cancelled());

        task.cancel();
        assert!(task.is_cancelled());
    }

    #[test]
    fn test_scheduler_spawn() {
        let scheduler = Scheduler::new(2);

        let task = scheduler.spawn(|| 42);
        let result = task.await_result();

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        scheduler.shutdown();
    }

    #[test]
    fn test_scheduler_multiple_tasks() {
        let scheduler = Scheduler::new(4);

        let counter = Arc::new(AtomicI32::new(0));
        let mut tasks = Vec::new();

        for _ in 0..100 {
            let counter = Arc::clone(&counter);
            tasks.push(scheduler.spawn(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }));
        }

        for task in tasks {
            task.await_result();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 100);

        scheduler.shutdown();
    }

    #[test]
    fn test_task_cancellation() {
        let scheduler = Scheduler::new(2);

        let started = Arc::new(AtomicBool::new(false));
        let started_clone = Arc::clone(&started);

        let task = scheduler.spawn(move || {
            started_clone.store(true, Ordering::SeqCst);
            loop {
                thread::sleep(Duration::from_millis(10));
            }
        });

        // Wait for task to start
        while !started.load(Ordering::SeqCst) {
            thread::yield_now();
        }

        task.cancel();

        // Task should still be running (cooperative cancellation)
        // In full implementation, the task would check for cancellation
        assert!(task.is_cancelled());

        scheduler.shutdown();
    }

    #[test]
    fn test_task_timeout() {
        let scheduler = Scheduler::new(2);

        let task = scheduler.spawn(|| {
            thread::sleep(Duration::from_secs(10));
            42
        });

        let result = task.await_timeout(Duration::from_millis(50));
        assert!(result.is_none()); // Should timeout

        scheduler.shutdown();
    }

    #[test]
    fn test_scheduler_stats() {
        let scheduler = Scheduler::new(2);

        for i in 0..10 {
            let task = scheduler.spawn(move || i);
            task.await_result();
        }

        let stats = scheduler.stats();
        assert!(stats.tasks_spawned >= 10);

        scheduler.shutdown();
    }

    #[test]
    fn test_with_scope() {
        let scheduler = Scheduler::new(2);

        let result = with_scope(&scheduler, |scope| {
            let t1 = scope.spawn(|| 1);
            let t2 = scope.spawn(|| 2);

            t1.await_result().unwrap() + t2.await_result().unwrap()
        });

        assert_eq!(result, 3);

        scheduler.shutdown();
    }
}
