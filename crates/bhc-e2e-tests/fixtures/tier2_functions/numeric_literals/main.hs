{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE BinaryLiterals #-}
module Main where

main :: IO ()
main = do
  -- Binary literals
  putStrLn (show (0b1010 :: Int))        -- 10
  putStrLn (show (0B11111111 :: Int))    -- 255

  -- Numeric underscores in decimal
  putStrLn (show (1_000_000 :: Int))     -- 1000000

  -- Numeric underscores in hex
  putStrLn (show (0xFF_FF :: Int))       -- 65535

  -- Numeric underscores in octal
  putStrLn (show (0o7_7_7 :: Int))       -- 511

  -- Numeric underscores in binary
  putStrLn (show (0b1111_0000 :: Int))   -- 240

  -- Arithmetic with these literals
  let x = 1_000 + 0b1010
  putStrLn (show x)                      -- 1010
