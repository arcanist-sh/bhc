//! WASI (WebAssembly System Interface) support.
//!
//! This module provides WASI imports and runtime functions needed for
//! standalone WASM execution with wasmtime, wasmer, or other WASI runtimes.

use crate::codegen::{WasmExport, WasmExportKind, WasmFunc, WasmFuncType, WasmGlobal, WasmImport, WasmImportKind};
use crate::{WasmInstr, WasmType};

/// Generate the standard WASI imports needed for basic I/O.
///
/// Returns a list of imports for:
/// - `fd_write`: Write to a file descriptor (used for stdout/stderr)
/// - `proc_exit`: Exit the process with a status code
pub fn generate_wasi_imports() -> Vec<WasmImport> {
    vec![
        // fd_write(fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) -> i32
        // Writes data to a file descriptor
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "fd_write".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(
                vec![WasmType::I32, WasmType::I32, WasmType::I32, WasmType::I32],
                vec![WasmType::I32],
            )),
        },
        // proc_exit(code: i32)
        // Terminates the process
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "proc_exit".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(vec![WasmType::I32], vec![])),
        },
        // fd_read(fd: i32, iovs: i32, iovs_len: i32, nread: i32) -> i32
        // Reads data from a file descriptor
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "fd_read".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(
                vec![WasmType::I32, WasmType::I32, WasmType::I32, WasmType::I32],
                vec![WasmType::I32],
            )),
        },
    ]
}

/// Index of fd_write in imports (assuming it's the first function import).
pub const FD_WRITE_IDX: u32 = 0;
/// Index of proc_exit in imports.
pub const PROC_EXIT_IDX: u32 = 1;
/// Index of fd_read in imports.
pub const FD_READ_IDX: u32 = 2;

/// Stdout file descriptor.
pub const STDOUT_FD: i32 = 1;
/// Stderr file descriptor.
pub const STDERR_FD: i32 = 2;

/// Generate the heap pointer global variable.
///
/// This global tracks the current end of the heap for allocation.
/// Initial value points after static data (e.g., at 64KB = 65536).
pub fn generate_heap_pointer_global() -> WasmGlobal {
    WasmGlobal {
        name: Some("heap_ptr".to_string()),
        ty: WasmType::I32,
        mutable: true,
        init: WasmInstr::I32Const(65536), // Start heap at 64KB
    }
}

/// Generate a simple bump allocator function.
///
/// This implements: `alloc(size: i32) -> i32`
/// Returns a pointer to allocated memory by bumping the heap pointer.
pub fn generate_alloc_function(heap_ptr_global: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![WasmType::I32]));
    func.name = Some("alloc".to_string());
    func.exported = true;
    func.export_name = Some("alloc".to_string());

    // Get current heap pointer
    func.emit(WasmInstr::GlobalGet(heap_ptr_global));

    // Duplicate for return value (local.tee pattern)
    let result_local = func.add_local(WasmType::I32);
    func.emit(WasmInstr::LocalTee(result_local));

    // Add size to heap pointer
    func.emit(WasmInstr::LocalGet(0)); // size parameter
    func.emit(WasmInstr::I32Add);

    // Align to 8 bytes: (ptr + 7) & ~7
    func.emit(WasmInstr::I32Const(7));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Const(-8)); // ~7 in two's complement
    func.emit(WasmInstr::I32And);

    // Store new heap pointer
    func.emit(WasmInstr::GlobalSet(heap_ptr_global));

    // Return original heap pointer
    func.emit(WasmInstr::LocalGet(result_local));
    func.emit(WasmInstr::End);

    func
}

/// Generate the print_i32 function for debugging.
///
/// Prints an i32 value to stdout using WASI fd_write.
/// Uses memory at a fixed offset for the iovec structure.
pub fn generate_print_i32(fd_write_idx: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![]));
    func.name = Some("print_i32".to_string());
    func.exported = true;

    // Convert i32 to decimal string in memory
    // Use fixed memory locations:
    // - 0-16: scratch space for number string
    // - 16-24: iovec structure

    let num_local = func.add_local(WasmType::I32); // number to print
    let ptr_local = func.add_local(WasmType::I32); // string pointer
    let len_local = func.add_local(WasmType::I32); // string length
    let digit_local = func.add_local(WasmType::I32);
    let is_neg_local = func.add_local(WasmType::I32);

    // Store parameter in local
    func.emit(WasmInstr::LocalGet(0));
    func.emit(WasmInstr::LocalSet(num_local));

    // Start at end of buffer (position 15)
    func.emit(WasmInstr::I32Const(15));
    func.emit(WasmInstr::LocalSet(ptr_local));

    // Initialize length to 0
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::LocalSet(len_local));

    // Check if negative
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32LtS);
    func.emit(WasmInstr::LocalSet(is_neg_local));

    // If negative, negate
    func.emit(WasmInstr::LocalGet(is_neg_local));
    func.emit(WasmInstr::If(None));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Sub);
    func.emit(WasmInstr::LocalSet(num_local));
    func.emit(WasmInstr::End);

    // Handle zero case
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Eqz);
    func.emit(WasmInstr::If(None));
    // Store '0' at position 15
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(48)); // '0'
    func.emit(WasmInstr::I32Store(1, 0));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::LocalSet(len_local));
    func.emit(WasmInstr::Else);

    // Convert digits loop
    func.emit(WasmInstr::Block(None)); // break target
    func.emit(WasmInstr::Loop(None)); // continue target

    // Get digit: num % 10
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Const(10));
    func.emit(WasmInstr::I32RemU);
    func.emit(WasmInstr::I32Const(48)); // '0'
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(digit_local));

    // Store digit
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::LocalGet(digit_local));
    func.emit(WasmInstr::I32Store(1, 0));

    // num = num / 10
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Const(10));
    func.emit(WasmInstr::I32DivU);
    func.emit(WasmInstr::LocalSet(num_local));

    // ptr--
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Sub);
    func.emit(WasmInstr::LocalSet(ptr_local));

    // len++
    func.emit(WasmInstr::LocalGet(len_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(len_local));

    // if num > 0, continue
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32GtU);
    func.emit(WasmInstr::BrIf(0)); // branch to loop

    func.emit(WasmInstr::End); // end loop
    func.emit(WasmInstr::End); // end block
    func.emit(WasmInstr::End); // end if (not zero)

    // Adjust ptr to point to start of string
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(ptr_local));

    // If negative, add '-' prefix
    func.emit(WasmInstr::LocalGet(is_neg_local));
    func.emit(WasmInstr::If(None));
    // Decrement ptr and store '-'
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Sub);
    func.emit(WasmInstr::LocalTee(ptr_local));
    func.emit(WasmInstr::I32Const(45)); // '-'
    func.emit(WasmInstr::I32Store(1, 0));
    func.emit(WasmInstr::LocalGet(len_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(len_local));
    func.emit(WasmInstr::End);

    // Set up iovec at memory offset 16
    // iovec.buf = ptr
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Store(4, 0));

    // iovec.len = len
    func.emit(WasmInstr::I32Const(20));
    func.emit(WasmInstr::LocalGet(len_local));
    func.emit(WasmInstr::I32Store(4, 0));

    // Call fd_write(stdout, iovec_ptr, 1, nwritten_ptr)
    func.emit(WasmInstr::I32Const(STDOUT_FD)); // fd = stdout
    func.emit(WasmInstr::I32Const(16)); // iovs = 16
    func.emit(WasmInstr::I32Const(1)); // iovs_len = 1
    func.emit(WasmInstr::I32Const(24)); // nwritten = 24
    func.emit(WasmInstr::Call(fd_write_idx));
    func.emit(WasmInstr::Drop); // ignore return value

    // Print newline
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32Const(10)); // '\n'
    func.emit(WasmInstr::I32Store(1, 0));
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32Store(4, 0));
    func.emit(WasmInstr::I32Const(20));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Store(4, 0));
    func.emit(WasmInstr::I32Const(STDOUT_FD));
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Const(24));
    func.emit(WasmInstr::Call(fd_write_idx));
    func.emit(WasmInstr::Drop);

    func.emit(WasmInstr::End);

    func
}

/// Generate a print_str function for printing string literals.
///
/// Takes a pointer and length, prints to stdout.
pub fn generate_print_str(fd_write_idx: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(
        vec![WasmType::I32, WasmType::I32], // ptr, len
        vec![],
    ));
    func.name = Some("print_str".to_string());
    func.exported = true;

    // Set up iovec at memory offset 16
    // iovec.buf = param 0 (ptr)
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::LocalGet(0)); // ptr
    func.emit(WasmInstr::I32Store(4, 0));

    // iovec.len = param 1 (len)
    func.emit(WasmInstr::I32Const(20));
    func.emit(WasmInstr::LocalGet(1)); // len
    func.emit(WasmInstr::I32Store(4, 0));

    // Call fd_write(stdout, iovec_ptr, 1, nwritten_ptr)
    func.emit(WasmInstr::I32Const(STDOUT_FD));
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Const(24));
    func.emit(WasmInstr::Call(fd_write_idx));
    func.emit(WasmInstr::Drop);

    func.emit(WasmInstr::End);

    func
}

/// Generate the _start function that calls main.
///
/// This is the WASI entry point. It calls the Haskell main function
/// and then calls proc_exit with 0.
pub fn generate_start_function(main_func_idx: u32, proc_exit_idx: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![]));
    func.name = Some("_start".to_string());
    func.exported = true;
    func.export_name = Some("_start".to_string());

    // Call main
    func.emit(WasmInstr::Call(main_func_idx));

    // Drop result if main returns something
    // (we assume main returns void or we ignore result)

    // Call proc_exit(0)
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::Call(proc_exit_idx));

    func.emit(WasmInstr::End);

    func
}

/// Generate a simple main that just returns 0.
///
/// This is a placeholder main function.
pub fn generate_placeholder_main() -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I32]));
    func.name = Some("main".to_string());

    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::End);

    func
}

/// Generate a main that prints "Hello, World!".
///
/// This is useful for testing the WASM pipeline.
pub fn generate_hello_main(print_str_idx: u32, string_offset: u32, string_len: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I32]));
    func.name = Some("main".to_string());

    // Call print_str(string_offset, string_len)
    func.emit(WasmInstr::I32Const(string_offset as i32));
    func.emit(WasmInstr::I32Const(string_len as i32));
    func.emit(WasmInstr::Call(print_str_idx));

    // Return 0
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::End);

    func
}

/// The "Hello, World!\n" string bytes.
pub const HELLO_WORLD_STRING: &[u8] = b"Hello, World!\n";

/// Offset where the hello world string is stored in memory.
pub const HELLO_WORLD_OFFSET: u32 = 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_wasi_imports() {
        let imports = generate_wasi_imports();
        assert_eq!(imports.len(), 3);
        assert_eq!(imports[0].name, "fd_write");
        assert_eq!(imports[1].name, "proc_exit");
        assert_eq!(imports[2].name, "fd_read");
    }

    #[test]
    fn test_generate_alloc() {
        let func = generate_alloc_function(0);
        assert_eq!(func.name.as_deref(), Some("alloc"));
        assert!(func.exported);
        assert!(!func.body.is_empty());
    }

    #[test]
    fn test_generate_print_i32() {
        let func = generate_print_i32(0);
        assert_eq!(func.name.as_deref(), Some("print_i32"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_start() {
        let func = generate_start_function(5, 1);
        assert_eq!(func.name.as_deref(), Some("_start"));
        assert!(func.exported);
        // Should contain call to main and proc_exit
        let calls: Vec<_> = func
            .body
            .iter()
            .filter(|i| matches!(i, WasmInstr::Call(_)))
            .collect();
        assert_eq!(calls.len(), 2);
    }
}
