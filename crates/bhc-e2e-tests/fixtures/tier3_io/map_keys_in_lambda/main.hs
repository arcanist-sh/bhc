-- Map and Set operations whose key is a lambda parameter: the key's type
-- is read from the operation's own type (#101).
module Main where

import Data.List (foldl')
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Text as T

data Color = Red | Green | Blue
  deriving (Show, Eq, Ord)

count :: [String] -> M.Map String Int
count = foldl' (\m w -> M.insertWith (+) w 1 m) M.empty

main :: IO ()
main = do
  let ws = words "a b a c a b"
  print (M.toList (foldl' (\m w -> M.insertWith (+) w 1 m) M.empty ws))
  print (M.toList (M.fromListWith (+) [(w, 1 :: Int) | w <- ws]))
  let m = count ws
  print (M.toList m)
  print (map (\w -> M.lookup w m) ["a", "c", "z"])
  print (map (\w -> M.member w m) ["b", "z"], map (\w -> M.findWithDefault 0 w m) ["a", "z"])
  print (M.toList (foldr (\w acc -> M.delete w acc) m ["a", "z"]))
  print (M.toList (foldl' (\acc w -> M.adjust (* 10) w acc) m ["b"]))
  print (M.toList (foldl' (\acc w -> M.alter (fmap (+ 1)) w acc) m ["c", "q"]))
  print (M.toList (foldl' (\acc (k, v) -> M.insert k v acc) M.empty (zip ws [1 :: Int ..])))
  let built = foldl' (\acc w -> S.insert w acc) S.empty ws
  print (S.size built, map (\w -> S.member w built) ["a", "z"])
  let s = foldr S.insert S.empty ws
  print (filter (\w -> S.member w s) ["a", "x", "c"])
  print (M.toList (foldl' (\acc t -> M.insertWith (+) t (1 :: Int) acc) M.empty (map T.pack ws)))
  print (M.toList (foldl' (\acc c -> M.insertWith (+) c (1 :: Int) acc) M.empty [Red, Blue, Red, Green, Blue, Red]))
  print (M.toList (foldl' (\acc n -> M.insertWith (+) n (1 :: Int) acc) M.empty [3, 1, 3, 2, 1, 3 :: Int]))
