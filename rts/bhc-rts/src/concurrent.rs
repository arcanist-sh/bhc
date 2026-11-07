//! `Control.Concurrent` for compiled Haskell: threads, `MVar`s and `Chan`s.
//!
//! Every Haskell thread is an OS thread, but only one runs at a time: it holds
//! a baton (`Sched::current`) and passes it on where it blocks, sleeps,
//! finishes, or has run for a time slice. Compiled code and thunk updates are
//! not safe to run in parallel, and the baton keeps them sequential, while
//! separate OS threads give each Haskell thread its own stack without any
//! context-switch code.
//!
//! A thread is preempted at an IO step (`safe_point`, which compiled code
//! reaches through `bhc_exception_check` at every `>>=`) once its slice is
//! over. A read from stdin gives the baton up while it waits
//! (`outside_baton`). When no thread can run, the baton is idle, and an
//! idle thread wakes sleepers and waits for readers to come back.
//!
//! ==== Limits
//!
//! * Pure code is not preempted: a thread in a long pure computation keeps
//!   the baton until its next IO step.
//! * An exception thrown to a thread waiting for stdin is raised when the read
//!   returns.
//! * `throwTo` and `killThread` reach a thread where it next runs: the
//!   other thread is always parked, blocked or waiting for the baton, so the
//!   exception is raised as it returns from there, and its handlers run.
//!   `mask` does not defer them.
//! * An RTS panic in a forked thread aborts the program. A thrown exception
//!   ends only that thread, and is reported on stderr as GHC does, except
//!   `ThreadKilled`.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The id of the program's main thread.
const MAIN_THREAD: u64 = 1;

/// `Sched::current` when no thread holds the baton.
const IDLE: u64 = 0;

/// How long a thread runs before it gives way at an IO step, like GHC's
/// default context-switch interval.
const TIME_SLICE: Duration = Duration::from_millis(20);

/// Stack of a forked thread. It is reserved, not committed, so the cost of an
/// idle thread is address space.
const FORKED_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// A thread waiting in an `MVar`'s queue.
struct Waiter {
    thread: u64,
    /// A `readMVar`, which leaves the value in place.
    is_read: bool,
}

/// An `MVar`. The object header makes it an evaluated heap object, so
/// `bhc_force` leaves it alone. Fields are only touched under the `SCHED` lock.
#[repr(C)]
pub struct MVar {
    tag: i64,
    full: bool,
    value: *mut u8,
    waiters: VecDeque<Waiter>,
    putters: VecDeque<(u64, *mut u8)>,
}

/// A `Chan`: an unbounded queue, with threads waiting when it is empty.
#[repr(C)]
pub struct Chan {
    tag: i64,
    items: VecDeque<*mut u8>,
    readers: VecDeque<u64>,
}

/// Where a parked thread waits, so an exception thrown to it can take it out.
#[derive(Clone, Copy)]
enum Blocked {
    Take(*mut MVar),
    Put(*mut MVar),
    ReadChan(*mut Chan),
    Sleep,
}

/// A `ThreadId`, with the header of an evaluated heap object.
#[repr(C)]
pub struct ThreadIdObject {
    tag: i64,
    id: u64,
}

struct Sched {
    /// The thread holding the baton.
    current: u64,
    next_id: u64,
    run_queue: VecDeque<u64>,
    sleepers: Vec<(Instant, u64)>,
    /// How a parked thread is woken, for every live thread.
    wakers: HashMap<u64, Arc<Condvar>>,
    /// Values handed to a thread that blocked in `takeMVar`/`readChan`.
    delivered: HashMap<u64, *mut u8>,
    /// Threads waiting in an `MVar`, a `Chan` or `threadDelay`.
    blocked: HashMap<u64, Blocked>,
    /// Exceptions thrown to a thread that has not run since, raised when it
    /// next gets the baton.
    interrupts: HashMap<u64, *mut u8>,
    /// Threads waiting for stdin without the baton.
    outside: usize,
}

// SAFETY: the raw pointers are Haskell values, which stay valid because the
// runtime does not free them, and only the baton holder uses them.
unsafe impl Send for Sched {}

static SCHED: Mutex<Option<Sched>> = Mutex::new(None);

/// Wakes the idle thread: the baton went idle, or a reader came back.
static IDLE_WAKER: Condvar = Condvar::new();

/// Set by the first `forkIO`. Until then there is one thread, and neither
/// preemption nor the stdin hand-off has anything to do.
static FORKED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static THIS_THREAD: Cell<u64> = const { Cell::new(MAIN_THREAD) };
    /// When this thread last got the baton.
    static SLICE_START: Cell<Option<Instant>> = const { Cell::new(None) };
}

type Guard = MutexGuard<'static, Option<Sched>>;

fn lock() -> Guard {
    let mut guard = SCHED.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let mut wakers = HashMap::new();
        wakers.insert(MAIN_THREAD, Arc::new(Condvar::new()));
        *guard = Some(Sched {
            current: MAIN_THREAD,
            next_id: MAIN_THREAD + 1,
            run_queue: VecDeque::new(),
            sleepers: Vec::new(),
            wakers,
            delivered: HashMap::new(),
            blocked: HashMap::new(),
            interrupts: HashMap::new(),
            outside: 0,
        });
    }
    guard
}

fn sched(guard: &mut Guard) -> &mut Sched {
    guard.as_mut().expect("scheduler is initialized by lock()")
}

/// Make every sleeper whose time has come runnable. Returns the earliest
/// wake-up time of those left.
fn wake_sleepers(s: &mut Sched) -> Option<Instant> {
    let now = Instant::now();
    let mut i = 0;
    while i < s.sleepers.len() {
        if s.sleepers[i].0 <= now {
            let (_, thread) = s.sleepers.remove(i);
            s.blocked.remove(&thread);
            s.run_queue.push_back(thread);
        } else {
            i += 1;
        }
    }
    s.sleepers.iter().map(|(at, _)| *at).min()
}

/// Every thread is blocked, so none can ever wake. GHC raises this in the
/// blocked main thread, and uncaught it ends the program the same way.
fn report_deadlock() -> ! {
    let _ = std::io::stdout().flush();
    let program = std::env::args().next().unwrap_or_else(|| "bhc".to_string());
    eprintln!("{program}: thread blocked indefinitely in an MVar operation");
    std::process::exit(1)
}

/// Pass the baton on, without waiting to get it back. With no thread ready, it
/// goes idle if a sleeper or a reader will want it later; otherwise every
/// thread is blocked for good.
fn hand_off(s: &mut Sched) {
    let earliest = wake_sleepers(s);
    match s.run_queue.pop_front() {
        Some(next) => {
            s.current = next;
            if let Some(waker) = s.wakers.get(&next) {
                waker.notify_one();
            }
        }
        None if earliest.is_some() || s.outside > 0 => {
            s.current = IDLE;
            IDLE_WAKER.notify_one();
        }
        None => report_deadlock(),
    }
}

/// The idle thread: while the baton is idle, wait for the earliest sleeper or
/// for a reader to come back, then pass the baton on.
fn run_idle() {
    let mut guard = lock();
    loop {
        let s = sched(&mut guard);
        if s.current != IDLE {
            guard = IDLE_WAKER.wait(guard).unwrap_or_else(|e| e.into_inner());
            continue;
        }
        let earliest = wake_sleepers(s);
        if !s.run_queue.is_empty() || (earliest.is_none() && s.outside == 0) {
            hand_off(s);
            continue;
        }
        guard = match earliest {
            Some(at) => {
                let wait = at.saturating_duration_since(Instant::now());
                IDLE_WAKER
                    .wait_timeout(guard, wait)
                    .unwrap_or_else(|e| e.into_inner())
                    .0
            }
            None => IDLE_WAKER.wait(guard).unwrap_or_else(|e| e.into_inner()),
        };
    }
}

/// Raise the exception thrown to this thread since it last ran, if any.
/// Returns whether there was one.
fn raise_interrupt(s: &mut Sched, me: u64) -> bool {
    match s.interrupts.remove(&me) {
        Some(exception) => {
            crate::ffi::raise_async(exception);
            true
        }
        None => false,
    }
}

/// A point where the running thread may be preempted: compiled code reaches
/// it at every IO step. Once the thread has had the baton for a time slice and
/// another thread is ready, it goes to the back of the queue. An exception
/// thrown to it meanwhile is raised here.
pub(crate) fn safe_point() {
    if !FORKED.load(Ordering::Relaxed) {
        return;
    }
    let started = SLICE_START.with(Cell::get);
    if started.is_some_and(|at| at.elapsed() < TIME_SLICE) {
        return;
    }
    let me = me();
    let mut guard = lock();
    let s = sched(&mut guard);
    wake_sleepers(s);
    if s.run_queue.is_empty() {
        SLICE_START.with(|cell| cell.set(Some(Instant::now())));
        return;
    }
    s.run_queue.push_back(me);
    let mut guard = park(guard, me);
    raise_interrupt(sched(&mut guard), me);
}

/// Run `read`, a read from stdin that may block, without the baton, so other
/// threads run while it waits. `read` must not touch Haskell values.
pub(crate) fn outside_baton<T>(read: impl FnOnce() -> T) -> T {
    if !FORKED.load(Ordering::Relaxed) {
        return read();
    }
    let me = me();
    let mut guard = lock();
    let s = sched(&mut guard);
    s.outside += 1;
    hand_off(s);
    drop(guard);
    let result = read();
    let mut guard = lock();
    let s = sched(&mut guard);
    s.outside -= 1;
    s.run_queue.push_back(me);
    if s.current == IDLE {
        IDLE_WAKER.notify_one();
    }
    let mut guard = wait_for_baton(guard, me);
    raise_interrupt(sched(&mut guard), me);
    result
}

/// Wait until this thread holds the baton.
fn wait_for_baton(mut guard: Guard, me: u64) -> Guard {
    let waker = sched(&mut guard)
        .wakers
        .get(&me)
        .cloned()
        .expect("a live thread has a waker");
    while sched(&mut guard).current != me {
        guard = waker.wait(guard).unwrap_or_else(|e| e.into_inner());
    }
    SLICE_START.with(|cell| cell.set(Some(Instant::now())));
    guard
}

/// Give up the baton until another thread makes this one runnable again.
fn park(mut guard: Guard, me: u64) -> Guard {
    hand_off(sched(&mut guard));
    wait_for_baton(guard, me)
}

/// Park this thread where `blocked` says until it is woken. Returns `false` if
/// it was woken by an exception thrown to it, which is then pending.
fn park_blocked(mut guard: Guard, me: u64, blocked: Blocked) -> (Guard, bool) {
    sched(&mut guard).blocked.insert(me, blocked);
    let mut guard = park(guard, me);
    let s = sched(&mut guard);
    s.blocked.remove(&me);
    if raise_interrupt(s, me) {
        s.delivered.remove(&me);
        return (guard, false);
    }
    (guard, true)
}

fn me() -> u64 {
    THIS_THREAD.with(Cell::get)
}

/// Make a blocked thread runnable.
fn wake(s: &mut Sched, thread: u64) {
    s.blocked.remove(&thread);
    s.run_queue.push_back(thread);
}

/// Give a blocked thread its value and make it runnable.
fn wake_with(s: &mut Sched, thread: u64, value: *mut u8) {
    s.delivered.insert(thread, value);
    wake(s, thread);
}

/// Run an IO action closure: force it, then call its code with the closure as
/// its environment. This is the closure layout codegen's `run_io_action` uses.
unsafe fn run_action(action: *mut u8) {
    unsafe {
        let action = crate::ffi::bhc_force(action);
        let code = *(action as *const extern "C" fn(*mut u8) -> *mut u8);
        code(action);
    }
}

/// A thin wrapper so an action pointer can move to the new OS thread. Only the
/// thread that receives the baton uses it.
struct SendPtr(*mut u8);
// SAFETY: see `SendPtr`.
unsafe impl Send for SendPtr {}

/// `forkIO`: start a thread running `action`. The caller keeps running.
///
/// # Safety
///
/// `action` must be an IO action closure or a thunk evaluating to one.
#[no_mangle]
pub unsafe extern "C" fn bhc_fork_io(action: *mut u8) -> *mut u8 {
    if !FORKED.swap(true, Ordering::Relaxed) {
        std::thread::Builder::new()
            .name("bhc-idle".to_string())
            .spawn(run_idle)
            .unwrap_or_else(|e| {
                eprintln!("forkIO: cannot create the idle thread: {e}");
                std::process::exit(1)
            });
        SLICE_START.with(|cell| cell.set(Some(Instant::now())));
    }
    let mut guard = lock();
    let s = sched(&mut guard);
    let id = s.next_id;
    s.next_id += 1;
    s.wakers.insert(id, Arc::new(Condvar::new()));
    s.run_queue.push_back(id);
    let action = SendPtr(action);
    std::thread::Builder::new()
        .stack_size(FORKED_THREAD_STACK_BYTES)
        .spawn(move || {
            let action = action;
            THIS_THREAD.with(|t| t.set(id));
            let mut guard = wait_for_baton(lock(), id);
            let interrupt = sched(&mut guard).interrupts.remove(&id);
            drop(guard);
            match interrupt {
                // Thrown to before it first ran: it ends at once.
                Some(exception) => crate::ffi::raise_async(exception),
                // SAFETY: the caller of `bhc_fork_io` guarantees the closure.
                None => unsafe { run_action(action.0) },
            }
            crate::ffi::report_thread_exception();
            let mut guard = lock();
            let s = sched(&mut guard);
            s.wakers.remove(&id);
            hand_off(s);
        })
        .unwrap_or_else(|e| {
            eprintln!("forkIO: cannot create a thread: {e}");
            std::process::exit(1)
        });
    drop(guard);
    new_thread_id(id)
}

fn new_thread_id(id: u64) -> *mut u8 {
    Box::into_raw(Box::new(ThreadIdObject { tag: 0, id })) as *mut u8
}

/// `myThreadId`.
#[no_mangle]
pub extern "C" fn bhc_my_thread_id() -> *mut u8 {
    new_thread_id(me())
}

/// `threadDelay`: sleep for at least `micros`, letting other threads run. A
/// delay of 0 gives way to every thread that is ready.
#[no_mangle]
pub extern "C" fn bhc_thread_delay(micros: i64) {
    if crate::ffi::exception_pending() {
        return;
    }
    let me = me();
    let mut guard = lock();
    let wake_at = Instant::now() + Duration::from_micros(micros.max(0).unsigned_abs());
    sched(&mut guard).sleepers.push((wake_at, me));
    drop(park_blocked(guard, me, Blocked::Sleep));
}

/// `throwTo`: raise `exception` in `thread`. Thrown to itself, it is a
/// `throwIO`. Any other live thread is parked, because only the caller runs:
/// a blocked one is taken out of its `MVar`, `Chan` or sleep and made
/// runnable, and it raises the exception when it next runs, so its handlers
/// run as in GHC. A finished thread ignores it.
///
/// # Safety
///
/// `thread` must be a pointer returned by `bhc_fork_io` or `bhc_my_thread_id`,
/// and `exception` an exception object.
#[no_mangle]
pub unsafe extern "C" fn bhc_throw_to(thread: *mut u8, exception: *mut u8) {
    if crate::ffi::exception_pending() {
        return;
    }
    let target = unsafe { (*(thread as *const ThreadIdObject)).id };
    if target == me() {
        crate::ffi::raise_async(exception);
        return;
    }
    let mut guard = lock();
    let s = sched(&mut guard);
    if !s.wakers.contains_key(&target) {
        return;
    }
    s.interrupts.entry(target).or_insert(exception);
    let Some(blocked) = s.blocked.remove(&target) else {
        // Runnable, or not started yet: it raises it when it gets the baton.
        return;
    };
    // SAFETY: a blocked thread's `MVar` or `Chan` stays allocated, and the
    // lock guards its fields.
    unsafe {
        match blocked {
            Blocked::Take(m) => (*m).waiters.retain(|w| w.thread != target),
            Blocked::Put(m) => (*m).putters.retain(|(t, _)| *t != target),
            Blocked::ReadChan(c) => (*c).readers.retain(|t| *t != target),
            Blocked::Sleep => s.sleepers.retain(|(_, t)| *t != target),
        }
    }
    s.run_queue.push_back(target);
}

/// `killThread`: `throwTo` with `ThreadKilled`.
///
/// # Safety
///
/// `thread` must be a pointer returned by `bhc_fork_io` or `bhc_my_thread_id`.
#[no_mangle]
pub unsafe extern "C" fn bhc_kill_thread(thread: *mut u8) {
    unsafe { bhc_throw_to(thread, crate::ffi::thread_killed()) }
}

/// `newEmptyMVar`.
#[no_mangle]
pub extern "C" fn bhc_new_empty_mvar() -> *mut u8 {
    Box::into_raw(Box::new(MVar {
        tag: 0,
        full: false,
        value: std::ptr::null_mut(),
        waiters: VecDeque::new(),
        putters: VecDeque::new(),
    })) as *mut u8
}

/// `newMVar`.
#[no_mangle]
pub extern "C" fn bhc_new_mvar(value: *mut u8) -> *mut u8 {
    let mvar = bhc_new_empty_mvar();
    // SAFETY: `mvar` was just allocated and nothing else can see it.
    unsafe {
        let m = &mut *(mvar as *mut MVar);
        m.full = true;
        m.value = value;
    }
    mvar
}

/// After the value left a full `MVar`: the first blocked `putMVar` fills it.
fn refill(s: &mut Sched, m: &mut MVar) {
    if let Some((thread, value)) = m.putters.pop_front() {
        m.full = true;
        m.value = value;
        wake(s, thread);
    }
}

/// `takeMVar` and `readMVar`.
unsafe fn take_or_read(mvar: *mut u8, is_read: bool) -> *mut u8 {
    if crate::ffi::exception_pending() {
        return std::ptr::null_mut();
    }
    let me = me();
    let mut guard = lock();
    // SAFETY: the caller passes an `MVar`, and the lock guards its fields.
    let m = unsafe { &mut *(mvar as *mut MVar) };
    let s = sched(&mut guard);
    if m.full {
        let value = m.value;
        if !is_read {
            m.full = false;
            m.value = std::ptr::null_mut();
            refill(s, m);
        }
        return value;
    }
    m.waiters.push_back(Waiter {
        thread: me,
        is_read,
    });
    let (mut guard, woken) = park_blocked(guard, me, Blocked::Take(m));
    if !woken {
        return std::ptr::null_mut();
    }
    sched(&mut guard)
        .delivered
        .remove(&me)
        .unwrap_or(std::ptr::null_mut())
}

/// `takeMVar`: empty the `MVar`, waiting until it is full.
///
/// # Safety
///
/// `mvar` must be a pointer returned by `bhc_new_mvar` or `bhc_new_empty_mvar`.
#[no_mangle]
pub unsafe extern "C" fn bhc_take_mvar(mvar: *mut u8) -> *mut u8 {
    unsafe { take_or_read(mvar, false) }
}

/// `readMVar`: the value of the `MVar`, waiting until it is full.
///
/// # Safety
///
/// `mvar` must be a pointer returned by `bhc_new_mvar` or `bhc_new_empty_mvar`.
#[no_mangle]
pub unsafe extern "C" fn bhc_read_mvar(mvar: *mut u8) -> *mut u8 {
    unsafe { take_or_read(mvar, true) }
}

/// Store into an empty `MVar`: every waiting `readMVar` sees the value, then
/// the first waiting `takeMVar` takes it, and otherwise it stays.
fn fill(s: &mut Sched, m: &mut MVar, value: *mut u8) {
    let mut taker = None;
    let mut rest = VecDeque::new();
    while let Some(waiter) = m.waiters.pop_front() {
        if waiter.is_read {
            wake_with(s, waiter.thread, value);
        } else if taker.is_none() {
            taker = Some(waiter.thread);
        } else {
            rest.push_back(waiter);
        }
    }
    m.waiters = rest;
    if let Some(thread) = taker {
        wake_with(s, thread, value);
    } else {
        m.full = true;
        m.value = value;
    }
}

/// `putMVar`: fill the `MVar`, waiting until it is empty.
///
/// # Safety
///
/// `mvar` must be a pointer returned by `bhc_new_mvar` or `bhc_new_empty_mvar`.
#[no_mangle]
pub unsafe extern "C" fn bhc_put_mvar(mvar: *mut u8, value: *mut u8) {
    if crate::ffi::exception_pending() {
        return;
    }
    let me = me();
    let mut guard = lock();
    // SAFETY: the caller passes an `MVar`, and the lock guards its fields.
    let m = unsafe { &mut *(mvar as *mut MVar) };
    if m.full {
        m.putters.push_back((me, value));
        drop(park_blocked(guard, me, Blocked::Put(m)));
    } else {
        fill(sched(&mut guard), m, value);
    }
}

/// A heap object: an ADT's tag followed by its fields.
fn new_object(tag: i64, fields: &[*mut u8]) -> *mut u8 {
    let mut words: Vec<usize> = Vec::with_capacity(1 + fields.len());
    words.push(tag as usize);
    words.extend(fields.iter().map(|f| *f as usize));
    Box::into_raw(words.into_boxed_slice()) as *mut usize as *mut u8
}

/// `tryTakeMVar`: `Just` the value if it was full, otherwise `Nothing`.
///
/// # Safety
///
/// `mvar` must be a pointer returned by `bhc_new_mvar` or `bhc_new_empty_mvar`.
#[no_mangle]
pub unsafe extern "C" fn bhc_try_take_mvar(mvar: *mut u8) -> *mut u8 {
    const NOTHING: i64 = 0;
    const JUST: i64 = 1;
    let mut guard = lock();
    // SAFETY: the caller passes an `MVar`, and the lock guards its fields.
    let m = unsafe { &mut *(mvar as *mut MVar) };
    if !m.full {
        return new_object(NOTHING, &[]);
    }
    let value = m.value;
    m.full = false;
    m.value = std::ptr::null_mut();
    refill(sched(&mut guard), m);
    new_object(JUST, &[value])
}

/// `tryPutMVar`: `True` if the `MVar` was empty and is filled now.
///
/// # Safety
///
/// `mvar` must be a pointer returned by `bhc_new_mvar` or `bhc_new_empty_mvar`.
#[no_mangle]
pub unsafe extern "C" fn bhc_try_put_mvar(mvar: *mut u8, value: *mut u8) -> *mut u8 {
    const FALSE: i64 = 0;
    const TRUE: i64 = 1;
    let mut guard = lock();
    // SAFETY: the caller passes an `MVar`, and the lock guards its fields.
    let m = unsafe { &mut *(mvar as *mut MVar) };
    if m.full {
        return new_object(FALSE, &[]);
    }
    fill(sched(&mut guard), m, value);
    new_object(TRUE, &[])
}

/// Refill an `MVar` a `modifyMVar_`/`withMVar` took from. If the action threw,
/// the old value goes back, so the exception does not leave the `MVar` empty.
///
/// # Safety
///
/// `mvar` must be a pointer returned by `bhc_new_mvar` or `bhc_new_empty_mvar`.
#[no_mangle]
pub unsafe extern "C" fn bhc_mvar_finish(mvar: *mut u8, old: *mut u8, new: *mut u8) {
    let value = if crate::ffi::exception_pending() {
        old
    } else {
        new
    };
    let mut guard = lock();
    // SAFETY: the caller passes an `MVar`, and the lock guards its fields.
    let m = unsafe { &mut *(mvar as *mut MVar) };
    fill(sched(&mut guard), m, value);
}

/// Finish `modifyMVar`: the action returned `pair`, `(new value, result)`. If
/// it threw, the old value goes back. Returns the result.
///
/// # Safety
///
/// `mvar` must be an `MVar`, and `pair` a pair unless an exception is pending.
#[no_mangle]
pub unsafe extern "C" fn bhc_mvar_finish_pair(
    mvar: *mut u8,
    old: *mut u8,
    pair: *mut u8,
) -> *mut u8 {
    if crate::ffi::exception_pending() {
        // SAFETY: the caller's guarantee for `mvar`.
        unsafe { bhc_mvar_finish(mvar, old, old) };
        return std::ptr::null_mut();
    }
    // SAFETY: a pair is an object with a one-word tag and two fields.
    let (new, result) = unsafe {
        let pair = crate::ffi::bhc_force(pair);
        (
            *(pair.add(8) as *mut *mut u8),
            *(pair.add(16) as *mut *mut u8),
        )
    };
    // SAFETY: the caller's guarantee for `mvar`.
    unsafe { bhc_mvar_finish(mvar, old, new) };
    result
}

/// `newChan`.
#[no_mangle]
pub extern "C" fn bhc_new_chan() -> *mut u8 {
    Box::into_raw(Box::new(Chan {
        tag: 0,
        items: VecDeque::new(),
        readers: VecDeque::new(),
    })) as *mut u8
}

/// `writeChan`: never blocks.
///
/// # Safety
///
/// `chan` must be a pointer returned by `bhc_new_chan`.
#[no_mangle]
pub unsafe extern "C" fn bhc_write_chan(chan: *mut u8, value: *mut u8) {
    if crate::ffi::exception_pending() {
        return;
    }
    let mut guard = lock();
    // SAFETY: the caller passes a `Chan`, and the lock guards its fields.
    let c = unsafe { &mut *(chan as *mut Chan) };
    let s = sched(&mut guard);
    match c.readers.pop_front() {
        Some(thread) => wake_with(s, thread, value),
        None => c.items.push_back(value),
    }
}

/// `readChan`: the oldest item, waiting until there is one.
///
/// # Safety
///
/// `chan` must be a pointer returned by `bhc_new_chan`.
#[no_mangle]
pub unsafe extern "C" fn bhc_read_chan(chan: *mut u8) -> *mut u8 {
    if crate::ffi::exception_pending() {
        return std::ptr::null_mut();
    }
    let me = me();
    let guard = lock();
    // SAFETY: the caller passes a `Chan`, and the lock guards its fields.
    let c = unsafe { &mut *(chan as *mut Chan) };
    if let Some(value) = c.items.pop_front() {
        return value;
    }
    c.readers.push_back(me);
    let (mut guard, woken) = park_blocked(guard, me, Blocked::ReadChan(c));
    if !woken {
        return std::ptr::null_mut();
    }
    sched(&mut guard)
        .delivered
        .remove(&me)
        .unwrap_or(std::ptr::null_mut())
}
