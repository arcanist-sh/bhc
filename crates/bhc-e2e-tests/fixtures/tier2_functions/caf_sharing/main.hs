-- A top-level constant (CAF) is computed once and shared by every use, as in
-- GHC: a global `IORef` made with `unsafePerformIO` is one `IORef`, and a
-- traced CAF traces once. An IO action defined at the top level still runs at
-- each use, and a CAF that throws throws again where it is used next.
import Debug.Trace (trace)
import Control.Exception
import Data.IORef
import System.IO.Unsafe (unsafePerformIO)

fib :: Int -> Int
fib k = if k < 2 then k else fib (k - 1) + fib (k - 2)

-- Counts how often `counted` is computed.
evaluations :: IORef Int
evaluations = unsafePerformIO (newIORef 0)
{-# NOINLINE evaluations #-}

counted :: Int
counted = unsafePerformIO (modifyIORef evaluations (+ 1) >> return (fib 25))
{-# NOINLINE counted #-}

big :: Int
big = trace "computing big" (fib 30)

squares :: [Int]
squares = map (\x -> x * x) [1 .. 10]

greeting :: String
greeting = "hello " ++ show big

hello :: IO ()
hello = putStrLn "hello from an IO CAF"

bad :: Int
bad = error "bad CAF"

zero :: Int
zero = 0

main :: IO ()
main = do
  print (counted + counted + counted)
  readIORef evaluations >>= print
  print big
  print (big * 2)
  print (sum squares, length squares)
  putStrLn greeting
  hello
  hello
  r1 <- try (evaluate bad)
  putStrLn (either (\e -> "caught: " ++ takeWhile (/= '\n') (show (e :: ErrorCall))) show r1)
  r2 <- try (evaluate bad)
  putStrLn (either (\e -> "caught again: " ++ takeWhile (/= '\n') (show (e :: ErrorCall))) show r2)
  print (zero, zero + 1)
