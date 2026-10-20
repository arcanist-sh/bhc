# Language Guide

bhc accepts Haskell 2010 plus a large part of the GHC extension surface. This
page is a short map of what you can rely on today; the complete reference is
at https://docs.arcanist.sh/bhc/. Every program below was compiled with bhc
v0.2.22 and a current development build, and its output matches GHC 9.8.2.

## Editions

The `--edition` flag is accepted, but it does not change behaviour yet
([#43](https://github.com/arcanist-sh/bhc/issues/43)). Extensions are enabled
with ordinary `LANGUAGE` pragmas.

## Core language

Functions, guards, `where` and `let`, `case`, lambdas, partial application,
algebraic data types, as-patterns, list comprehensions, and `do` notation in
`IO` and the list monad all work as in GHC:

```haskell
module Main where

classify :: Int -> String
classify n
  | n < 0     = "negative"
  | n == 0    = "zero"
  | otherwise = "positive"

quadratic :: Int -> Int -> Int -> Int -> Int
quadratic a b c x = term1 + term2 + c
  where
    term1 = a * x * x
    term2 = b * x

describe :: Maybe Int -> String
describe mx = case mx of
  Nothing -> "no value"
  Just n  -> "value: " ++ show n

safeDiv :: Int -> Int -> Maybe Int
safeDiv _ 0 = Nothing
safeDiv x y = Just (x `div` y)

pairs :: [a] -> [b] -> [(a, b)]
pairs xs ys = do
  x <- xs
  y <- ys
  return (x, y)

data Tree a = Leaf | Node (Tree a) a (Tree a)

insert :: Ord a => a -> Tree a -> Tree a
insert x Leaf = Node Leaf x Leaf
insert x t@(Node l v r)
  | x < v     = Node (insert x l) v r
  | x > v     = Node l v (insert x r)
  | otherwise = t

toList :: Tree a -> [a]
toList Leaf         = []
toList (Node l v r) = toList l ++ [v] ++ toList r

main :: IO ()
main = do
  mapM_ (putStrLn . classify) [-3, 0, 7]
  print (quadratic 1 2 3 4)
  putStrLn (describe (safeDiv 10 2))
  putStrLn (describe (safeDiv 1 0))
  print (pairs [1, 2] [10, 20 :: Int])
  print (toList (foldr insert Leaf [5, 3, 8, 1, 4, 7 :: Int]))
  print (map add5 [1, 2, 3])

add :: Int -> Int -> Int
add x y = x + y

-- Partial application
add5 :: Int -> Int
add5 = add 5
```

Output:

```
negative
zero
positive
27
value: 5
no value
[(1,10),(1,20),(2,10),(2,20)]
[1,3,4,5,7,8]
[6,7,8]
```

## Modules, imports, strict fields, Text

Qualified and hiding imports, `Data.Map`, `Data.List`, `Data.Text` (including
`OverloadedStrings` literals at `Text`), strictness annotations, and `INLINE`
pragmas:

```haskell
{-# LANGUAGE OverloadedStrings #-}
module Main where

import Prelude hiding (lookup)
import qualified Data.Map as M
import Data.List (sort, nub)
import qualified Data.Text as T

{-# INLINE dbl #-}
dbl :: Int -> Int
dbl x = x * 2

data P = P !Int !Int

normSq :: P -> Int
normSq (P x y) = x * x + y * y

dup :: [a] -> [a]
dup list@(x:_) = x : list
dup []         = []

lookup :: Int -> M.Map Int String -> String
lookup k m = M.findWithDefault "?" k m

main :: IO ()
main = do
  print (normSq (P 3 4))
  print (dup [7, 8 :: Int])
  print (sort (nub [3, 1, 3, 2 :: Int]))
  putStrLn (lookup 2 (M.fromList [(1, "one"), (2, "two")]))
  print (dbl (dbl 5))
  putStrLn (T.unpack (T.toUpper (T.pack "text")))
  print (T.length "hello")
```

Output:

```
25
[7,7,8]
[1,2,3]
two
20
TEXT
5
```

## Type classes

User-defined classes, constrained instances (`instance C a => C [a]`) and
superclasses are compiled with dictionary passing, and `deriving` works for the
stock classes; the type-class and data-type programs in
[Examples](examples.md) use both. One rough edge: `show` reached through a
polymorphic path, such as a class default method or a list built with
`[minBound .. maxBound]`, can print raw pointers instead of constructor names
([#24](https://github.com/arcanist-sh/bhc/issues/24)). Derived `show` on a
record declared with field syntax shows the same problem; give such types a
hand-written rendering function for now.

## Extensions

Common extensions such as `LambdaCase`, `MultiWayIf`, `TupleSections`,
`RecordWildCards`, `NamedFieldPuns`, `BangPatterns`, `ScopedTypeVariables` and
`OverloadedStrings` work in compiled programs; the GHC-extensions program in
[Examples](examples.md) exercises several of them. Pandoc's 221 modules, which
use a wide range of extensions, all pass `bhc check` and compile.

Known gaps in the type-level extensions:

- GADTs with more than one index ([#33](https://github.com/arcanist-sh/bhc/issues/33))
- associated type family instances used at a literal ([#34](https://github.com/arcanist-sh/bhc/issues/34))
- `DerivingVia` ([#35](https://github.com/arcanist-sh/bhc/issues/35)) and
  `UnboxedTuples` ([#36](https://github.com/arcanist-sh/bhc/issues/36))
- kind-annotated type parameters ([#31](https://github.com/arcanist-sh/bhc/issues/31))

## Pragmas

- `LANGUAGE` pragmas enable extensions.
- `INLINE`, `NOINLINE` and `INLINABLE` are parsed and recorded in interfaces.
- User `RULES` pragmas are not applied.
- There is no `OPTIONS_BHC` pragma and there are no bhc-specific language
  extensions. Profiles are chosen per build with `--profile` or in `hx.toml`
  (see [Profiles](profiles.md)).

## Evaluation

Compiled code is lazy under every profile, so infinite lists, `takeWhile` over
`[1..]`, and knot-tying work as in GHC. Bang patterns and strict fields force
evaluation where you ask for it.

## Library gaps to know about

Some library functions still give wrong results; the list with issue numbers
is at the end of [Examples](examples.md) and on the
[issue tracker](https://github.com/arcanist-sh/bhc/issues). The most common in
small programs are `Double` arithmetic in folds and fields (#18), `getLine`
(#20), `Maybe`/`Either` as monads (#23), and `show` for some tuples (#24).

## See also

- [Getting Started](getting-started.md)
- [Examples](examples.md)
- [Profiles](profiles.md)
