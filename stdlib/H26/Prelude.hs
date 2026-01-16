-- |
-- Module      : H26.Prelude
-- Description : Standard Haskell 2026 Prelude
-- License     : BSD-3-Clause
--
-- The H26 Prelude re-exports commonly used types and functions.
-- This is automatically imported unless explicitly hidden.

{-# HASKELL_EDITION 2026 #-}

module H26.Prelude
  ( -- * Basic types
    Bool(..)
  , Maybe(..)
  , Either(..)
  , Ordering(..)
  , Char
  , String
  , Int
  , Integer
  , Float
  , Double

    -- * Basic type classes
  , Eq(..)
  , Ord(..)
  , Show(..)
  , Read(..)
  , Enum(..)
  , Bounded(..)
  , Num(..)
  , Integral(..)
  , Fractional(..)
  , Floating(..)
  , Real(..)
  , RealFrac(..)
  , RealFloat(..)

    -- * Functor hierarchy
  , Functor(..)
  , Applicative(..)
  , Monad(..)
  , (>>=)
  , (>>)
  , return

    -- * Foldable and Traversable
  , Foldable(..)
  , Traversable(..)

    -- * Monoid
  , Semigroup(..)
  , Monoid(..)

    -- * Common functions
  , id
  , const
  , (.)
  , flip
  , ($)
  , ($!)
  , (&)
  , undefined
  , error
  , seq
  , fst
  , snd
  , curry
  , uncurry

    -- * List operations
  , map
  , (++)
  , filter
  , head
  , tail
  , last
  , init
  , null
  , length
  , (!!)
  , reverse
  , take
  , drop
  , splitAt
  , takeWhile
  , dropWhile
  , span
  , break
  , elem
  , notElem
  , lookup
  , zip
  , zip3
  , zipWith
  , zipWith3
  , unzip
  , unzip3

    -- * Folds
  , foldl
  , foldl'
  , foldl1
  , foldr
  , foldr1
  , sum
  , product
  , maximum
  , minimum
  , concat
  , concatMap
  , and
  , or
  , any
  , all

    -- * Boolean
  , (&&)
  , (||)
  , not
  , otherwise

    -- * Maybe
  , maybe
  , fromMaybe
  , isJust
  , isNothing

    -- * Either
  , either
  , isLeft
  , isRight

    -- * Tuples
  , swap

    -- * Numeric
  , subtract
  , even
  , odd
  , gcd
  , lcm
  , (^)
  , (^^)
  , fromIntegral
  , realToFrac

    -- * IO
  , IO
  , putChar
  , putStr
  , putStrLn
  , print
  , getChar
  , getLine
  , readIO
  , readLn

    -- * Files
  , FilePath
  , readFile
  , writeFile
  , appendFile

    -- * H26 Extensions
  , lazy
  ) where

-- This is a specification file.
-- Actual implementation provided by the compiler.
