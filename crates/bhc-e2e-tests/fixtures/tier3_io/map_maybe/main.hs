module Main where

import qualified Data.Map as Map

safeDiv :: Int -> Maybe Int
safeDiv 0 = Nothing
safeDiv n = Just (100 `div` n)

main :: IO ()
main = do
  let m = Map.fromList [(1, 5), (2, 0), (3, 10), (4, 0), (5, 2)]
  let result = Map.mapMaybe safeDiv m
  putStrLn (show (Map.size result))
  putStrLn (show (Map.elems result))
