-- `show` of a Maybe returned by a call, by `lookup`/`find`/`elemIndex`, or by
-- `fmap` over one, renders its payload by type: a String quoted, a negative
-- number parenthesised, a Double and a Bool as themselves.
import Data.List (find, elemIndex)

half :: Int -> Maybe Int
half n = if even n then Just (n `div` 2) else Nothing

neg :: Int -> Maybe Int
neg n = Just (negate n)

ratio :: Double -> Maybe Double
ratio d = if d > 0 then Just (d / 2) else Nothing

isBig :: Int -> Maybe Bool
isBig n = Just (n > 10)

main :: IO ()
main = do
  print (half 10)
  print (half 3)
  print (fmap (+ 1) (half 4))
  print ((* 3) <$> half 6)
  print (lookup 2 [(1, "a"), (2, "b")])
  print (lookup 3 [(1, "a"), (2, "b")])
  print (lookup 'y' [('x', 1), ('y', 2 :: Int)])
  print (neg 4)
  print (ratio 5)
  print (isBig 20)
  print (find even [1, 3, 4, 5 :: Int])
  print (elemIndex 'c' "abcd")
