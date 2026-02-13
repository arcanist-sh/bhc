{-# LANGUAGE OverloadedStrings #-}
module Main where

main :: IO ()
main = do
  let x = "hello" :: String
  putStrLn x
  putStrLn (fromString "world")
  let y = fromString "test" :: String
  putStrLn y
