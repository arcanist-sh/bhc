//! A read from stdin lets other threads run while it waits (#75).
//!
//! A fixture's stdin is written all at once, so it cannot show this: here the
//! program's stdin stays open until another thread has printed, which it can
//! only do if `getLine` gave the baton up.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const PROGRAM: &str = "\
import Control.Concurrent
import System.IO

main :: IO ()
main = do
  hSetBuffering stdout NoBuffering
  done <- newEmptyMVar
  _ <- forkIO $ do
    threadDelay 10000
    putStrLn \"ticker ran\"
    putMVar done ()
  l <- getLine
  takeMVar done
  putStrLn (\"got \" ++ l)
";

#[test]
fn get_line_lets_other_threads_run() {
    // Threads run only under the closure IO model.
    if std::env::var("BHC_IO_MODEL").as_deref() == Ok("eager") {
        return;
    }
    let work = tempfile::tempdir().expect("temporary directory");
    let source = work.path().join("Main.hs");
    std::fs::write(&source, PROGRAM).expect("write the program");
    let binary = work.path().join("main");
    let compile = Command::new("cargo")
        .args(["run", "--quiet", "-p", "bhc", "--"])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("run bhc");
    assert!(
        compile.status.success(),
        "bhc failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start the program");
    let stdout = child.stdout.take().expect("stdout");
    let (lines_tx, lines) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if lines_tx.send(line).is_err() {
                break;
            }
        }
    });

    let first = lines.recv_timeout(Duration::from_secs(30));
    let mut stdin = child.stdin.take().expect("stdin");
    let _ = stdin.write_all(b"hi\n");
    drop(stdin);
    let status = child.wait().expect("wait for the program");
    assert_eq!(
        first.as_deref(),
        Ok("ticker ran"),
        "the other thread did not run while getLine waited"
    );
    let second = lines.recv_timeout(Duration::from_secs(30));
    assert_eq!(second.as_deref(), Ok("got hi"));
    assert!(status.success(), "exit status {status}");
}
