-- An explicit `;` separates declarations inside a layout block: a one-line
-- `let`, a `where` helper's `let`, and an instance body. Only the
-- layout-inserted separator was accepted, so the `;` ended the enclosing
-- declaration (`unbound variable: b`), and an instance written this way
-- inside a lambda was dropped whole.
module Main where

class Two a where
  one :: a -> Int
  two :: a -> Int

data T = T

instance Two T where one _ = 1; two _ = 2

f :: Int -> Int -> Int
f a b = let x = a + 1; y = b + 2 in x * y

g :: Int -> Int -> Int
g a b = h 0
  where
    h z = let x = a + z; y = b + 2 in x * y

main :: IO ()
main = do
  print (f 1 2)
  print (g 1 2)
  print (one T + two T)
