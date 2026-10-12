-- A failing guard in a `case` alternative falls through to the NEXT
-- alternative (it used to raise "Non-exhaustive patterns").
firstIs :: Char -> String -> Maybe Char
firstIs c s = case s of
  (x:_) | x == c -> Just x
  _ -> Nothing

classify :: Int -> String
classify n = case n of
  0 -> "zero"
  k | k < 0 -> "negative"
    | k > 100 -> "large"
  k | even k -> "even"
  _ -> "odd"

lookupPair :: Maybe (Int, Int) -> Int
lookupPair m = case m of
  Just (a, b) | a > b -> a
              | b > 10 -> b
  Just (a, _) -> a * 100
  Nothing -> -1

-- A guarded `True` before `False`: the resume case has no `True` arm.
sign :: Bool -> Int -> Int
sign b n = case b of
  True | n > 0 -> 1
       | n < 0 -> 2
  False -> 3

out :: Maybe Char -> String
out Nothing = "Nothing"
out (Just ch) = "Just " ++ [ch]

main :: IO ()
main = do
  putStrLn (out (firstIs 'a' "ab"))
  putStrLn (out (firstIs 'x' "ab"))
  mapM_ (putStrLn . classify) [0, -3, 200, 8, 7]
  print (lookupPair (Just (5, 2)))
  print (lookupPair (Just (1, 20)))
  print (lookupPair (Just (3, 4)))
  print (lookupPair Nothing)
  print [sign True 5, sign True (-1), sign False 0]
