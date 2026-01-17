-- Test: type-inference
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 4.1

{-# HASKELL_EDITION 2026 #-}

module TypeInferenceTest where

-- Let-polymorphism: id should have type forall a. a -> a
test1 :: (Int, Bool)
test1 =
  let id = \x -> x  -- Inferred: forall a. a -> a
  in (id 42, id True)

-- Monomorphism restriction does NOT apply by default in H26
-- This should be polymorphic
test2 :: (Int, Double)
test2 =
  let f = (+)  -- Inferred: forall a. Num a => a -> a -> a
  in (f 1 2, f 1.0 2.0)

-- Type inference with constraints
test3 :: Int
test3 =
  let sum xs = foldr (+) 0 xs  -- Inferred: forall a. Num a => [a] -> a
  in sum [1, 2, 3]

-- Higher-rank types require annotation
test4 :: Int
test4 =
  let apply :: (forall a. a -> a) -> (Int, Bool)
      apply f = (f 1, f True)
  in fst (apply id)

-- Existential types in data
data Showable = forall a. Show a => MkShowable a

test5 :: String
test5 =
  let x = MkShowable 42
      showIt (MkShowable a) = show a
  in showIt x

-- Record types with inference
data Person = Person { name :: String, age :: Int }

test6 :: String
test6 =
  let p = Person { name = "Alice", age = 30 }
  in name p  -- Field accessor has type Person -> String

-- GADTs require type signature
data Expr a where
  LitInt :: Int -> Expr Int
  LitBool :: Bool -> Expr Bool
  Add :: Expr Int -> Expr Int -> Expr Int

-- Evaluator must have explicit type
eval :: Expr a -> a
eval (LitInt n) = n
eval (LitBool b) = b
eval (Add x y) = eval x + eval y

test7 :: Int
test7 = eval (Add (LitInt 1) (LitInt 2))
