# Examples

This page contains example programs demonstrating BHC's features.

## Hello World

The classic first program:

```haskell
module Main where

main :: IO ()
main = putStrLn "Hello, World!"
```

```bash
bhc hello.hs -o hello && ./hello
```

## Fibonacci

Recursive fibonacci with pattern matching:

```haskell
module Main where

fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)

main :: IO ()
main = print (fib 20)
```

## Factorial with Guards

Using guards for conditional logic:

```haskell
module Main where

factorial :: Integer -> Integer
factorial n
  | n < 0     = error "Negative factorial"
  | n == 0    = 1
  | otherwise = n * factorial (n - 1)

main :: IO ()
main = print (factorial 20)
```

## List Processing

Working with lists using higher-order functions:

```haskell
module Main where

-- Sum of squares of even numbers
sumEvenSquares :: [Int] -> Int
sumEvenSquares xs = sum (map (^2) (filter even xs))

-- Alternative with function composition
sumEvenSquares' :: [Int] -> Int
sumEvenSquares' = sum . map (^2) . filter even

main :: IO ()
main = do
  let numbers = [1..10]
  print $ sumEvenSquares numbers   -- 220
  print $ sumEvenSquares' numbers  -- 220
```

## Numeric Profile: Dot Product

High-performance numeric code with guaranteed fusion:

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
module DotProduct where

-- Fuses into a single SIMD loop
dotProduct :: [Double] -> [Double] -> Double
dotProduct xs ys = sum (zipWith (*) xs ys)

-- Manual benchmark
main :: IO ()
main = do
  let n = 1000000
      xs = replicate n 1.0
      ys = replicate n 2.0
  print $ dotProduct xs ys  -- 2000000.0
```

Verify fusion:

```bash
bhc --profile=numeric --kernel-report dotproduct.hs
# [Kernel k1] dotProduct: FUSED
#   Pattern: sum/zipWith
#   SIMD width: 8 x f64 (AVX-512)
```

## Numeric Profile: Matrix Multiply

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
module MatMul where

import BHC.Tensor

-- Matrix multiplication (fused and tiled)
matmul :: Tensor '[M, K] Double -> Tensor '[K, N] Double -> Tensor '[M, N] Double
matmul a b = T.contract a b

main :: IO ()
main = do
  let a = T.randn [256, 256]
      b = T.randn [256, 256]
  print $ T.shape (matmul a b)  -- [256, 256]
```

## Server Profile: HTTP Handler

Concurrent request handling with structured concurrency:

```haskell
{-# OPTIONS_BHC -profile=server #-}
module Server where

import Control.Concurrent.Scope
import Control.Monad (forM)

-- Handle request with timeout
handleRequest :: Request -> IO Response
handleRequest req = withDeadline (seconds 30) $ \scope -> do
    -- Fetch data in parallel
    user <- spawn scope $ fetchUser (reqUserId req)
    posts <- spawn scope $ fetchPosts (reqUserId req)
    friends <- spawn scope $ fetchFriends (reqUserId req)

    -- Await all results
    u <- await user
    p <- await posts
    f <- await friends

    pure $ Response
      { respUser = u
      , respPosts = p
      , respFriends = f
      }

-- Fan-out pattern
fetchAll :: [UserId] -> IO [User]
fetchAll ids = withScope $ \scope -> do
    tasks <- forM ids $ \id -> spawn scope (fetchUser id)
    mapM await tasks
```

## Server Profile: Producer-Consumer

Using STM for coordination:

```haskell
{-# OPTIONS_BHC -profile=server #-}
module ProducerConsumer where

import Control.Concurrent.STM
import Control.Concurrent.Scope

type Queue a = TVar [a]

newQueue :: IO (Queue a)
newQueue = newTVarIO []

enqueue :: Queue a -> a -> STM ()
enqueue q x = modifyTVar' q (++ [x])

dequeue :: Queue a -> STM a
dequeue q = do
    xs <- readTVar q
    case xs of
        []     -> retry  -- Block until item available
        (x:xs') -> do
            writeTVar q xs'
            pure x

main :: IO ()
main = withScope $ \scope -> do
    queue <- newQueue

    -- Producer
    producer <- spawn scope $ do
        mapM_ (\x -> atomically (enqueue queue x)) [1..100]

    -- Consumer
    consumer <- spawn scope $ do
        replicateM_ 100 $ do
            x <- atomically (dequeue queue)
            print x

    await producer
    await consumer
```

## Edge Profile: WASM Function

Minimal WebAssembly export:

```haskell
{-# OPTIONS_BHC -profile=edge #-}
module WasmExport where

-- Exported to WASM
foreign export ccall "add" add :: Int -> Int -> Int

add :: Int -> Int -> Int
add x y = x + y

-- Entry point for standalone WASM
main :: IO ()
main = print (add 1 2)
```

Build for WASM:

```bash
bhc --target=wasi --profile=edge wasm.hs -o app.wasm
wasmtime app.wasm
```

## Realtime Profile: Game Loop

Bounded GC pauses for smooth rendering:

```haskell
{-# OPTIONS_BHC -profile=realtime #-}
module Game where

import BHC.Realtime.Arena

data GameState = GameState
  { gsPlayer :: !Vec2
  , gsEnemies :: ![Enemy]
  , gsScore :: !Int
  }

gameLoop :: GameState -> IO ()
gameLoop state = do
    -- Frame arena: all allocations freed at frame end
    withFrameArena $ do
        input <- pollInput

        -- Update (allocates in frame arena)
        let state' = updateGame state input

        -- Render (allocates in frame arena)
        renderGame state'

        -- Loop
        unless (shouldQuit input) $
            gameLoop state'

updateGame :: GameState -> Input -> GameState
updateGame state input =
    state { gsPlayer = movePlayer (gsPlayer state) input
          , gsEnemies = map updateEnemy (gsEnemies state)
          }
```

## Custom Data Types

Algebraic data types with pattern matching:

```haskell
module Main where

-- Sum type
data Shape
  = Circle Double
  | Rectangle Double Double
  | Triangle Double Double Double
  deriving (Show)

-- Calculate area
area :: Shape -> Double
area (Circle r) = pi * r * r
area (Rectangle w h) = w * h
area (Triangle a b c) =
    let s = (a + b + c) / 2
    in sqrt (s * (s - a) * (s - b) * (s - c))

main :: IO ()
main = do
    let shapes = [Circle 5, Rectangle 4 6, Triangle 3 4 5]
    mapM_ (\s -> print (s, area s)) shapes
```

## Type Classes

Defining and implementing type classes:

```haskell
module Main where

-- Define a serialization type class
class Serialize a where
    serialize :: a -> String
    deserialize :: String -> Maybe a

-- Implement for Int
instance Serialize Int where
    serialize = show
    deserialize s = case reads s of
        [(n, "")] -> Just n
        _         -> Nothing

-- Implement for Bool
instance Serialize Bool where
    serialize True = "true"
    serialize False = "false"
    deserialize "true" = Just True
    deserialize "false" = Just False
    deserialize _ = Nothing

-- Implement for lists
instance Serialize a => Serialize [a] where
    serialize xs = "[" ++ intercalate "," (map serialize xs) ++ "]"
    deserialize = parseList

main :: IO ()
main = do
    print $ serialize (42 :: Int)
    print $ serialize [1, 2, 3 :: Int]
```

## Monads: State

Using the State monad:

```haskell
module Main where

import Control.Monad.State

type Counter = State Int

increment :: Counter ()
increment = modify (+1)

getCount :: Counter Int
getCount = get

runCounter :: Counter a -> Int -> (a, Int)
runCounter = runState

main :: IO ()
main = do
    let (result, finalCount) = runCounter computation 0
    print finalCount
  where
    computation = do
        increment
        increment
        increment
        n <- getCount
        increment
        pure n

-- Output: 4 (returned 3, then incremented to 4)
```

## IO Operations

File handling and command-line arguments:

```haskell
module Main where

import System.Environment (getArgs)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [inputFile, outputFile] -> do
            content <- readFile inputFile
            let processed = processContent content
            writeFile outputFile processed
            putStrLn $ "Processed " ++ inputFile ++ " -> " ++ outputFile

        _ -> putStrLn "Usage: program <input> <output>"

processContent :: String -> String
processContent = unlines . map processLine . lines

processLine :: String -> String
processLine = map toUpper
```

## See Also

- [Getting Started](getting-started.md) - Installation guide
- [Language Guide](language.md) - Language reference
- [Profiles](profiles.md) - Profile documentation
