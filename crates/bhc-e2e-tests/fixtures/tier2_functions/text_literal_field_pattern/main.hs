-- A Text string literal matched against a constructor field inside a local
-- function without a signature compares as Text. The field binder was
-- untyped, so codegen compared the Text field against a list literal and
-- never matched; pandoc's `taskListItemFromAscii` (`fromMd (Str "[" : …)`)
-- never recognized a task list.
{-# LANGUAGE OverloadedStrings #-}
module Main where
import qualified Data.Text as T
import Data.Text (Text)

data Inl = Str Text | Space
data Blk = Plain [Inl] | Para [Inl]

fromAscii :: Bool -> [Blk] -> [Blk]
fromAscii = handle fromMd
  where
    fromMd (Str "[" : Space : Str "]" : Space : is) = Str "BOX" : Space : is
    fromMd (Str "[x]" : Space : is) = Str "XBOX" : Space : is
    fromMd is = is

handle :: ([Inl] -> [Inl]) -> Bool -> [Blk] -> [Blk]
handle handleInlines on bls =
  if on
  then handleItem bls
  else bls
  where
    handleItem (Plain is : bs) = Plain (handleInlines is) : bs
    handleItem (Para is : bs) = Para (handleInlines is) : bs
    handleItem bs = bs

render :: [Blk] -> String
render (Plain is : _) = concatMap r is
  where
    r (Str t) = T.unpack t
    r Space = " "
render _ = "other"

main :: IO ()
main = do
  putStrLn (render (fromAscii True [Plain [Str "[", Space, Str "]", Space, Str "task"]]))
  putStrLn (render (fromAscii True [Plain [Str "[x]", Space, Str "done"]]))
