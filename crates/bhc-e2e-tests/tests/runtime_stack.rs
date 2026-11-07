//! Deep evaluation runs on a large stack, and running out of it is reported
//! instead of crashing (#82).
//!
//! Both programs are compiled at `-O0`, where demand analysis does not make
//! the accumulator strict, so the first builds a long chain of thunks.

use std::process::{Command, Output};

/// Compile `source` at `-O0` and run it.
fn compile_and_run(source: &str) -> Output {
    let work = tempfile::tempdir().expect("temporary directory");
    let path = work.path().join("Main.hs");
    std::fs::write(&path, source).expect("write the program");
    let binary = work.path().join("main");
    let compile = Command::new("cargo")
        .args(["run", "--quiet", "-p", "bhc", "--", "-O0"])
        .arg(&path)
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("run bhc");
    assert!(
        compile.status.success(),
        "bhc failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    Command::new(&binary).output().expect("run the program")
}

#[test]
fn long_thunk_chain_fits_on_the_main_stack() {
    let output = compile_and_run(concat!(
        "sumTo :: Int -> Double -> Double\n",
        "sumTo 0 acc = acc\n",
        "sumTo n acc = sumTo (n - 1) (acc + fromIntegral n)\n",
        "\n",
        "main :: IO ()\n",
        "main = print (sumTo 2000000 0)\n",
    ));
    assert!(output.status.success(), "status {}", output.status);
    assert_eq!(String::from_utf8_lossy(&output.stdout), "2.000001e12\n");
}

#[test]
fn stack_overflow_is_reported() {
    let output = compile_and_run(concat!(
        "deep :: Int -> Int\n",
        "deep 0 = 0\n",
        "deep n = 1 + deep (n - 1)\n",
        "\n",
        "main :: IO ()\n",
        "main = print (deep 1000000000)\n",
    ));
    assert_eq!(output.status.code(), Some(2), "status {}", output.status);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.ends_with(": stack overflow\n"), "stderr: {stderr:?}");
}
