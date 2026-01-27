# Language Guide

BHC implements Haskell 2026 (H26) with compatibility modes for earlier standards.

## Editions

| Edition | Flag | Description |
|---------|------|-------------|
| Haskell 2010 | `--edition=Haskell2010` | Standard Haskell 2010 |
| GHC2021 | `--edition=GHC2021` | GHC 2021 defaults |
| GHC2024 | `--edition=GHC2024` | GHC 2024 defaults |
| H26 | `--edition=H26` | Haskell 2026 (default) |

## Basic Syntax

### Comments

```haskell
-- Single line comment

{- Multi-line
   comment -}

-- | Documentation comment
-- Appears in generated docs
```

### Literals

```haskell
-- Integers
42
-17
0x2A      -- Hexadecimal
0o52      -- Octal
0b101010  -- Binary

-- Floating point
3.14
2.5e10
1e-6

-- Characters
'a'
'\n'      -- Newline
'\x41'    -- Unicode

-- Strings
"Hello, World!"
"Line 1\nLine 2"

-- Lists
[1, 2, 3]
[1..10]           -- [1,2,3,4,5,6,7,8,9,10]
[1,3..10]         -- [1,3,5,7,9]
[x | x <- [1..10], even x]  -- [2,4,6,8,10]

-- Tuples
(1, "hello")
(1, 2, 3)
```

### Functions

```haskell
-- Function definition
add :: Int -> Int -> Int
add x y = x + y

-- Pattern matching
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Guards
classify :: Int -> String
classify n
  | n < 0     = "negative"
  | n == 0    = "zero"
  | otherwise = "positive"

-- Where clause
quadratic :: Double -> Double -> Double -> Double -> Double
quadratic a b c x = a * x * x + b * x + c
  where
    term1 = a * x * x
    term2 = b * x

-- Let expression
quadratic' a b c x =
  let term1 = a * x * x
      term2 = b * x
  in term1 + term2 + c

-- Lambda
double :: Int -> Int
double = \x -> x * 2

-- Partial application
add5 :: Int -> Int
add5 = add 5
```

### Operators

```haskell
-- Arithmetic
(+), (-), (*), (/)
div, mod           -- Integer division
(^)                -- Power

-- Comparison
(==), (/=)
(<), (>), (<=), (>=)

-- Logical
(&&), (||), not

-- List
(:)                -- Cons
(++)               -- Append
(!!)               -- Index

-- Function
(.)                -- Composition
($)                -- Application
```

## Type System

### Basic Types

```haskell
-- Primitive types
Int        -- Fixed-precision integer
Integer    -- Arbitrary-precision integer
Float      -- Single-precision float
Double     -- Double-precision float
Bool       -- True | False
Char       -- Unicode character

-- Common types
String     -- [Char]
[a]        -- List of a
Maybe a    -- Nothing | Just a
Either a b -- Left a | Right b
(a, b)     -- Tuple
```

### Type Signatures

```haskell
-- Simple signature
double :: Int -> Int
double x = x * 2

-- Polymorphic signature
identity :: a -> a
identity x = x

-- Constrained signature
sum :: Num a => [a] -> a
sum = foldl (+) 0

-- Multiple constraints
showAndCompare :: (Show a, Ord a) => a -> a -> String
showAndCompare x y
  | x < y     = show x ++ " < " ++ show y
  | otherwise = show x ++ " >= " ++ show y
```

### Algebraic Data Types

```haskell
-- Sum type (alternatives)
data Bool = False | True

data Maybe a = Nothing | Just a

data Either a b = Left a | Right b

-- Product type (records)
data Person = Person
  { name :: String
  , age :: Int
  , email :: String
  }

-- Mixed
data Tree a
  = Leaf a
  | Branch (Tree a) (Tree a)
```

### Type Aliases

```haskell
type String = [Char]

type Name = String

type Point = (Double, Double)

type Predicate a = a -> Bool
```

### Newtypes

```haskell
newtype Age = Age Int

newtype Email = Email String
  deriving (Show, Eq)
```

## Type Classes

### Common Type Classes

```haskell
-- Equality
class Eq a where
  (==) :: a -> a -> Bool
  (/=) :: a -> a -> Bool

-- Ordering
class Eq a => Ord a where
  compare :: a -> a -> Ordering
  (<), (>), (<=), (>=) :: a -> a -> Bool

-- Show (convert to string)
class Show a where
  show :: a -> String

-- Read (parse from string)
class Read a where
  read :: String -> a

-- Numeric
class Num a where
  (+), (-), (*) :: a -> a -> a
  negate, abs, signum :: a -> a
  fromInteger :: Integer -> a

-- Functor
class Functor f where
  fmap :: (a -> b) -> f a -> f b

-- Applicative
class Functor f => Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

-- Monad
class Applicative m => Monad m where
  (>>=) :: m a -> (a -> m b) -> m b
  return :: a -> m a
```

### Defining Instances

```haskell
data Color = Red | Green | Blue

instance Eq Color where
  Red == Red = True
  Green == Green = True
  Blue == Blue = True
  _ == _ = False

instance Show Color where
  show Red = "Red"
  show Green = "Green"
  show Blue = "Blue"

-- Derived instances
data Point = Point Double Double
  deriving (Eq, Show, Ord)
```

## Pattern Matching

### Basic Patterns

```haskell
-- Literal patterns
isZero :: Int -> Bool
isZero 0 = True
isZero _ = False

-- Constructor patterns
describe :: Maybe Int -> String
describe Nothing = "no value"
describe (Just n) = "value: " ++ show n

-- List patterns
head :: [a] -> a
head (x:_) = x
head [] = error "empty list"

-- Tuple patterns
fst :: (a, b) -> a
fst (x, _) = x

-- As-patterns
dup :: [a] -> [a]
dup list@(x:_) = x : list
dup [] = []
```

### Case Expressions

```haskell
describe :: Maybe Int -> String
describe mx = case mx of
  Nothing -> "no value"
  Just n  -> "value: " ++ show n
```

### Pattern Guards

```haskell
lookup' :: Eq k => k -> [(k, v)] -> Maybe v
lookup' key pairs
  | Just v <- lookup key pairs = Just v
  | otherwise = Nothing
```

## Monads and Do Notation

### Maybe Monad

```haskell
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

calculate :: Double -> Double -> Double -> Maybe Double
calculate x y z = do
  a <- safeDivide x y
  b <- safeDivide a z
  pure (b + 1)
```

### List Monad

```haskell
pairs :: [a] -> [b] -> [(a, b)]
pairs xs ys = do
  x <- xs
  y <- ys
  pure (x, y)

-- Equivalent to:
pairs' xs ys = [(x, y) | x <- xs, y <- ys]
```

### IO Monad

```haskell
main :: IO ()
main = do
  putStrLn "What's your name?"
  name <- getLine
  putStrLn ("Hello, " ++ name ++ "!")
```

## Modules

### Module Declaration

```haskell
module Data.MyList
  ( MyList(..)      -- Export type and all constructors
  , singleton       -- Export function
  , fromList        -- Export function
  ) where
```

### Imports

```haskell
-- Import everything
import Data.List

-- Import specific items
import Data.List (sort, nub)

-- Qualified import
import qualified Data.Map as Map

-- Hiding imports
import Prelude hiding (map, filter)
```

## Extensions

### Language Pragmas

```haskell
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
```

### Common Extensions

| Extension | Description |
|-----------|-------------|
| `OverloadedStrings` | String literals as any `IsString` |
| `LambdaCase` | `\case` syntax |
| `MultiWayIf` | Multi-way if expressions |
| `RecordWildCards` | `{..}` in patterns |
| `NamedFieldPuns` | `{field}` shorthand |
| `TupleSections` | `(,x)` for partial tuples |
| `GADTs` | Generalized algebraic data types |
| `TypeFamilies` | Type-level functions |
| `DataKinds` | Promote types to kinds |
| `RankNTypes` | Higher-rank polymorphism |

## BHC-Specific Features

### Profile Selection

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
module FastMath where
```

### Fusion Hints

```haskell
{-# INLINE myFunction #-}
{-# NOINLINE expensiveFunction #-}
{-# RULES "map/map" forall f g xs. map f (map g xs) = map (f . g) xs #-}
```

### Strictness Annotations

```haskell
data Strict = Strict !Int !String

-- Strict field
data Point = Point { x :: !Double, y :: !Double }
```

## See Also

- [Getting Started](getting-started.md) - Installation and first steps
- [Profiles](profiles.md) - Runtime profiles
- [Examples](examples.md) - Code examples
