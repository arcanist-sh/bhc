-- A value that matches no clause, or no alternative of a case, is a pattern
-- match failure. The compiler warned that the patterns were incomplete and
-- the program went on with a wrong value: the last clause was run for a
-- value it did not match, so `f (C 7)` answered as `f B`, `k 3` as `k 2`,
-- and `g Nothing` read a field `Nothing` does not have.
import Control.Exception

data T = A | B | C Int deriving Show

f :: T -> Int
f A = 1
f B = 2

g :: Maybe Int -> Int
g (Just x) = x

h :: [Int] -> Int
h (x : _) = x

k :: Int -> String
k 1 = "one"
k 2 = "two"

c :: Char -> Int
c 'a' = 1

s :: String -> Int
s "yes" = 1
s "no" = 0

report :: String -> Either SomeException String -> IO ()
report what (Left _) = putStrLn (what ++ ": pattern match failure")
report what (Right v) = putStrLn (what ++ ": " ++ v)

main :: IO ()
main = do
  print (f A, f B, g (Just 5), h [9, 8], k 2, c 'a', s "no")
  r1 <- try (evaluate (show (f (C 7))))
  report "f (C 7)" r1
  r2 <- try (evaluate (show (g Nothing)))
  report "g Nothing" r2
  r3 <- try (evaluate (show (h [])))
  report "h []" r3
  r4 <- try (evaluate (k 3))
  report "k 3" r4
  r5 <- try (evaluate (show (c 'b')))
  report "c 'b'" r5
  r6 <- try (evaluate (show (s "maybe")))
  report "s \"maybe\"" r6
  r7 <- try (evaluate (show (case B of A -> 0 :: Int)))
  report "case B of A" r7
  -- A complete match is unaffected.
  print (map f [A, B], case C 4 of { C n -> n; _ -> 0 }, [x | Just x <- [Just 1, Nothing, Just (3 :: Int)]])
