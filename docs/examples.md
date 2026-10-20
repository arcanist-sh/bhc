# Examples

Each program on this page was compiled with bhc v0.2.22 (release) and a current
development build, run, and compared byte-for-byte with GHC 9.8.2 (`runghc`).
The output shown is the real output. More examples live in the repository's
[`examples/`](../examples/) folder and in the
[documentation](https://docs.arcanist.sh/bhc/).

Compile and run any of them with:

```bash
bhc Example.hs -o example && ./example
```

## Hello World

```haskell
main = putStrLn "Hello, World!"
```

Output:

```
Hello, World!
```

## Fibonacci

Recursion and pattern matching on `Int`.

```haskell
module Main where

fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)

main :: IO ()
main = print (fib 20)
```

Output:

```
6765
```

## Factorial with guards

`factorial 20` still fits in a 64-bit `Int`.

```haskell
module Main where

factorial :: Int -> Int
factorial n
  | n < 0     = error "Negative factorial"
  | n == 0    = 1
  | otherwise = n * factorial (n - 1)

main :: IO ()
main = print (factorial 20)
```

Output:

```
2432902008176640000
```

## List processing

Higher-order functions, composition, comprehensions, and an infinite list consumed lazily.

```haskell
module Main where

square :: Int -> Int
square x = x * x

-- Sum of squares of even numbers
sumEvenSquares :: [Int] -> Int
sumEvenSquares xs = sum (map square (filter even xs))

-- The same, as a composition
sumEvenSquares' :: [Int] -> Int
sumEvenSquares' = sum . map square . filter even

main :: IO ()
main = do
  let numbers = [1..10]
  print (sumEvenSquares numbers)
  print (sumEvenSquares' numbers)
  print [x | x <- numbers, even x]
  print (takeWhile (< 10) [1..])
```

Output:

```
220
220
[2,4,6,8,10]
[1,2,3,4,5,6,7,8,9]
```

## Data types

A sum type with derived `Show`, consumed by pattern matching.

```haskell
module Main where

data Shape
  = Square Int
  | Rectangle Int Int
  | Triangle Int Int
  deriving (Show)

-- Area, doubled so it stays an Int
area2 :: Shape -> Int
area2 (Square s)      = 2 * s * s
area2 (Rectangle w h) = 2 * w * h
area2 (Triangle b h)  = b * h

describe :: Shape -> String
describe s = show s ++ " has twice-area " ++ show (area2 s)

main :: IO ()
main = mapM_ (putStrLn . describe) [Square 5, Rectangle 4 6, Triangle 3 4]
```

Output:

```
Square 5 has twice-area 50
Rectangle 4 6 has twice-area 48
Triangle 3 4 has twice-area 12
```

## Type classes

A user-defined class with instances for base types, a constrained instance for lists, and a derived `Ord`.

```haskell
module Main where

import Data.List (intercalate)

class Serialize a where
  serialize :: a -> String

instance Serialize Int where
  serialize n = show n

instance Serialize Bool where
  serialize True  = "true"
  serialize False = "false"

instance Serialize a => Serialize [a] where
  serialize xs = "[" ++ intercalate "," (map serialize xs) ++ "]"

data Color = Red | Green | Blue
  deriving (Show, Eq, Ord)

instance Serialize Color where
  serialize Red   = "red"
  serialize Green = "green"
  serialize Blue  = "blue"

main :: IO ()
main = do
  putStrLn (serialize (42 :: Int))
  putStrLn (serialize [1, 2, 3 :: Int])
  putStrLn (serialize [True, False])
  putStrLn (serialize [Red, Green, Blue])
  print (Red < Blue, Green == Green)
```

Output:

```
42
[1,2,3]
[true,false]
[red,green,blue]
(True,True)
```

## Records and Data.Map

Record construction, field access and update, and a `Data.Map` with `Int` keys.

```haskell
module Main where

import qualified Data.Map as Map

data Person = Person
  { name :: String
  , age  :: Int
  }

people :: [Person]
people = [Person "Ada" 36, Person "Alan" 41, Person "Grace" 45]

describe :: Person -> String
describe p = name p ++ " (" ++ show (age p) ++ ")"

-- How many people per decade of age
perDecade :: [Person] -> Map.Map Int Int
perDecade ps = Map.fromListWith (+) [(age p `div` 10 * 10, 1) | p <- ps]

main :: IO ()
main = do
  mapM_ (putStrLn . describe) people
  let older = map (\p -> p { age = age p + 1 }) people
  print (map age older)
  print (Map.toList (perDecade people))
  print (Map.lookup 40 (perDecade people))
```

Output:

```
Ada (36)
Alan (41)
Grace (45)
[37,42,46]
[(30,1),(40,2)]
Just 2
```

## State monad

`Control.Monad.State` from the bundled transformers library.

```haskell
module Main where

import Control.Monad.State

type Counter = State Int

increment :: Counter ()
increment = modify (+1)

computation :: Counter Int
computation = do
  increment
  increment
  increment
  n <- get
  increment
  return n

main :: IO ()
main = do
  let (result, finalCount) = runState computation 0
  print result
  print finalCount
```

Output:

```
3
4
```

## StateT over IO

A monad transformer stack that interleaves state with output.

```haskell
module Main where

import Control.Monad.State

tick :: String -> StateT Int IO ()
tick label = do
  n <- get
  lift (putStrLn (label ++ " at " ++ show n))
  put (n + 1)

pipeline :: StateT Int IO ()
pipeline = do
  tick "load"
  tick "parse"
  tick "emit"

main :: IO ()
main = do
  final <- execStateT pipeline 0
  print final
```

Output:

```
load at 0
parse at 1
emit at 2
3
```

## GHC extensions

A handful of the extensions bhc supports.

```haskell
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE BangPatterns #-}
module Main where

data Config = Config { host :: String, port :: Int }

render :: Config -> String
render Config{..} = host ++ ":" ++ show port

portOf :: Config -> Int
portOf Config{port} = port

sign :: Int -> String
sign n = if | n < 0 -> "-" | n == 0 -> "0" | otherwise -> "+"

describe :: Maybe Int -> String
describe = \case
  Nothing -> "none"
  Just n  -> "just " ++ show n

sumStrict :: [Int] -> Int
sumStrict = go 0
  where
    go !acc []     = acc
    go !acc (x:xs) = go (acc + x) xs

main :: IO ()
main = do
  let cfg = Config { host = "localhost", port = 8080 }
  putStrLn (render cfg)
  print (portOf cfg)
  putStrLn (concatMap sign [-2, 0, 5])
  mapM_ (putStrLn . describe) [Nothing, Just 3]
  print (map (,0) [1, 2 :: Int])
  print (sumStrict [1 .. 100])
```

Output:

```
localhost:8080
8080
-0+
none
just 3
[(1,0),(2,0)]
5050
```

## File IO

`readFile`/`writeFile` and string processing.

```haskell
module Main where

import Data.Char (toUpper)

main :: IO ()
main = do
  writeFile "input.txt" "hello\nfrom bhc\n"
  content <- readFile "input.txt"
  let processed = unlines (map (map toUpper) (lines content))
  writeFile "output.txt" processed
  result <- readFile "output.txt"
  putStr result
  print (length (lines result), length (words result))
```

Output:

```
HELLO
FROM BHC
(2,3)
```

The same program runs on WebAssembly. WASI only lets the program see
directories you pass with `--dir`:

```bash
bhc --target wasm32-wasi Upper.hs -o upper.wasm
wasmtime --dir=. upper.wasm
```

It prints the same three lines and leaves `input.txt` and `output.txt` in the
current directory.

## Numeric profile: fused folds

Under `--profile numeric`, bhc rewrites `sum`/`product`/`foldl'` over an `Int`
range (optionally through a `map` of a named `Int -> Int` function), `sum (map f xs)` for such an `f`, and `map f (map g xs)` into single traversals before native
code generation (see [Profiles](profiles.md)).

```haskell
module Main where

import Data.List (foldl')

double :: Int -> Int
double x = 2 * x

main :: IO ()
main = do
  print (sum (map double [1 .. 1000000]))
  print (foldl' (+) 0 (map double (map double [1 .. 1000])))
```

```bash
bhc --profile numeric SumSq.hs -o sumsq && ./sumsq
```

Output:

```
1000001000000
2002000
```

`--kernel-report` prints the Tensor IR fusion report. List code like this is
fused by the Core rewrites instead, so the report shows zero tensor operations
for it.

## Things to avoid in examples for now

A few library paths still give wrong answers or abort at run time. The open
issues are tracked at https://github.com/arcanist-sh/bhc/issues; the ones most
likely to bite small programs are:

- `Double` arithmetic on constructor fields, `sum`, and `foldr` ([#18](https://github.com/arcanist-sh/bhc/issues/18)); `pi` is unbound ([#19](https://github.com/arcanist-sh/bhc/issues/19))
- `getLine` ([#20](https://github.com/arcanist-sh/bhc/issues/20)) and `getEnv` ([#21](https://github.com/arcanist-sh/bhc/issues/21))
- `sum` over `Integer` ([#22](https://github.com/arcanist-sh/bhc/issues/22))
- `Applicative`/`Monad` for `Maybe` and `Either`, including `do` blocks in them ([#23](https://github.com/arcanist-sh/bhc/issues/23))
- `show` for 3-tuples and for tuples containing `Bool` ([#24](https://github.com/arcanist-sh/bhc/issues/24))
- `Data.Map` with `String` keys ordering ([#25](https://github.com/arcanist-sh/bhc/issues/25)) and `Data.Map.Strict` ([#26](https://github.com/arcanist-sh/bhc/issues/26))
- `read` at `Double` or list types ([#29](https://github.com/arcanist-sh/bhc/issues/29))
- point-free instance methods ([#30](https://github.com/arcanist-sh/bhc/issues/30)) and derived `Functor`/`Traversable` ([#32](https://github.com/arcanist-sh/bhc/issues/32))
- `forkIO` and `MVar` ([#38](https://github.com/arcanist-sh/bhc/issues/38))

## See also

- [Getting Started](getting-started.md)
- [Language Guide](language.md)
- [Profiles](profiles.md)
