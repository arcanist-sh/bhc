//! Stacks for Haskell threads, and reporting a stack overflow (#82).
//!
//! Compiled code evaluates on the C stack: forcing a chain of thunks recurses
//! once per link. The OS main thread has a small stack (8 MB on macOS), so
//! [`bhc_run_main`] runs the program on a thread with [`MAIN_STACK_BYTES`]
//! reserved. The reservation is address space; pages are committed as the
//! stack grows.
//!
//! A thread that still runs out hits the guard page below its stack. Instead of
//! a bare segfault, a signal handler running on an alternate stack reports it
//! as GHC does, `prog: stack overflow`, and exits with status 2. A fault
//! anywhere else is left to the default action.
//!
//! ==== Limits
//!
//! * Output the program buffered but has not flushed yet is lost when it
//!   overflows: the handler cannot safely take the stdout lock.
//! * An overflow in a forked thread ends the program. GHC raises
//!   `StackOverflow` in that thread only.

use std::cell::Cell;
use std::sync::OnceLock;

/// Stack reserved for the thread running `main`.
pub const MAIN_STACK_BYTES: usize = 1 << 30;

/// Size of each thread's alternate signal stack, which the overflow handler
/// runs on.
const SIGNAL_STACK_BYTES: usize = 64 * 1024;

/// How close to the low end of a thread's stack a fault must be to count as
/// an overflow. The estimate of the low end is off by the frames above the
/// point where it was taken, and a large frame can skip past the guard page.
const OVERFLOW_WINDOW_BYTES: usize = 1024 * 1024;

thread_local! {
    /// The estimated lowest address of this thread's stack, or 0 when it is
    /// not known (threads not started through [`enter_thread`]).
    static STACK_LOW: Cell<usize> = const { Cell::new(0) };
}

/// `prog: stack overflow\n`, prepared before any overflow, since the handler
/// may not allocate.
static OVERFLOW_MESSAGE: OnceLock<Vec<u8>> = OnceLock::new();

/// Record the stack of the thread that is starting, `stack_bytes` in size,
/// and give it an alternate signal stack. Call it first thing in the thread.
pub(crate) fn enter_thread(stack_bytes: usize) {
    let marker = 0u8;
    let top = std::ptr::addr_of!(marker) as usize;
    STACK_LOW.with(|low| low.set(top.saturating_sub(stack_bytes)));
    install_signal_stack();
}

/// Give this thread an alternate stack for signal handlers. The memory is
/// never freed: threads that exit are few, and the handler may need it until
/// the very end.
fn install_signal_stack() {
    let memory: &'static mut [u8] = Box::leak(vec![0u8; SIGNAL_STACK_BYTES].into_boxed_slice());
    let stack = libc::stack_t {
        ss_sp: memory.as_mut_ptr().cast(),
        ss_flags: 0,
        ss_size: SIGNAL_STACK_BYTES,
    };
    // SAFETY: `stack` describes memory that lives for the rest of the process.
    unsafe {
        libc::sigaltstack(&stack, std::ptr::null_mut());
    }
}

/// Install the overflow handler for `SIGSEGV` and `SIGBUS`, once.
fn install_handler() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let program = std::env::args().next().unwrap_or_else(|| "bhc".to_string());
        let _ = OVERFLOW_MESSAGE.set(format!("{program}: stack overflow\n").into_bytes());
        for signal in [libc::SIGSEGV, libc::SIGBUS] {
            // SAFETY: a zeroed `sigaction` is a valid starting point; the
            // handler has the `SA_SIGINFO` signature.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = on_fault as *const () as usize;
                action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(signal, &action, std::ptr::null_mut());
            }
        }
    });
}

/// The faulting address of a `SIGSEGV` or `SIGBUS`.
///
/// # Safety
///
/// `info` must be the `siginfo_t` the kernel passed to the handler.
unsafe fn fault_address(info: *const libc::siginfo_t) -> usize {
    #[cfg(target_os = "linux")]
    {
        unsafe { (*info).si_addr() as usize }
    }
    #[cfg(not(target_os = "linux"))]
    {
        unsafe { (*info).si_addr as usize }
    }
}

/// Signal handler: report a stack overflow and exit, or let any other fault
/// take its default action.
extern "C" fn on_fault(signal: libc::c_int, info: *mut libc::siginfo_t, _: *mut libc::c_void) {
    // SAFETY: the kernel passes a valid `siginfo_t`.
    let address = unsafe { fault_address(info) };
    let low = STACK_LOW.with(Cell::get);
    let overflow = low != 0 && address.abs_diff(low) <= OVERFLOW_WINDOW_BYTES;
    if overflow {
        if let Some(message) = OVERFLOW_MESSAGE.get() {
            // SAFETY: `write` and `_exit` are async-signal-safe.
            unsafe {
                libc::write(2, message.as_ptr().cast(), message.len());
            }
        }
        // SAFETY: async-signal-safe; status 2, as GHC exits on a stack overflow.
        unsafe { libc::_exit(2) }
    }
    // Not an overflow: restore the default action, and returning re-runs the
    // faulting instruction, which then crashes as it would have.
    // SAFETY: resetting a signal to its default action.
    unsafe {
        libc::signal(signal, libc::SIG_DFL);
    }
}

/// Run the program's `main`, `body`, on a thread with a large stack, and
/// return when it does. Compiled programs call this from the C `main` after
/// `bhc_rts_init`.
///
/// # Safety
///
/// `body` must be the program's compiled main body.
#[no_mangle]
pub unsafe extern "C" fn bhc_run_main(body: extern "C" fn()) {
    install_handler();
    let runner = std::thread::Builder::new()
        .name("main".to_string())
        .stack_size(MAIN_STACK_BYTES)
        .spawn(move || {
            enter_thread(MAIN_STACK_BYTES);
            body();
        });
    match runner {
        Ok(handle) => {
            if handle.join().is_err() {
                std::process::exit(1);
            }
        }
        // Without a thread to run it on, run it here, on the C stack.
        Err(_) => body(),
    }
}
