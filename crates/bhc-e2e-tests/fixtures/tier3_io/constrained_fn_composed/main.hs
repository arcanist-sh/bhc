{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
-- Test: a constrained function used point-free whose class argument
-- appears only in its constraint
-- Verifies: `length . describe` passes describe its dictionary although the
-- element type `e` is fixed only through the functional dependency
-- (pandoc's markdown `endline` writes `notFollowedBy . char` this way).

class Container f e | f -> e where
  firstOf :: f -> Maybe e
  render :: e -> String

instance Container [Char] Char where
  firstOf (c : _) = Just c
  firstOf [] = Nothing
  render c = [c, c]

describe :: Container f e => f -> String
describe xs = maybe "none" render (firstOf xs)

main :: IO ()
main = do
  putStrLn (describe "abc")
  print ((length . describe) "xyz")
  putStrLn (concatMap (reverse . describe) ["pq", ""])
