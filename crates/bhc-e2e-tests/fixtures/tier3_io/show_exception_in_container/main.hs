-- Exceptions shown inside Either, lists, tuples and Maybe (#103).
module Main where

import Control.Exception

safeDiv :: Int -> Int -> IO (Either ArithException Int)
safeDiv a b = try (evaluate (a `div` b))

main :: IO ()
main = do
  r <- try (evaluate (1 `div` (0 :: Int))) :: IO (Either ArithException Int)
  print r
  case r of
    Left e -> putStrLn ("caught: " ++ show e)
    Right v -> print v
  r2 <- try (evaluate (5 `mod` (0 :: Int))) :: IO (Either ArithException Int)
  print r2
  r3 <- try (evaluate (10 `div` (2 :: Int))) :: IO (Either ArithException Int)
  print r3
  r6 <- try (ioError (userError "bad")) :: IO (Either IOException ())
  print r6
  rs <- mapM (safeDiv 12) [3, 0, 4]
  print rs
  print (either show show r, fmap (+ 1) r3)
  a <- safeDiv 1 0
  print (a, 7 :: Int)
  print (either Just (const Nothing) a)
  handle (\e -> putStrLn ("handled: " ++ show (e :: ArithException))) (evaluate (1 `div` (0 :: Int)) >>= print)
