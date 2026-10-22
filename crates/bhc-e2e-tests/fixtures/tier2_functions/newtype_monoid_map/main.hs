-- A newtype over a Map deriving Semigroup/Monoid (pandoc's `Translations`).
-- `mempty`/`<>` at the newtype are the Map's: newtypes are erased, so a
-- bare builtin `mempty` walked as a Map crashed the first lookup.
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

import qualified Data.Map as M

newtype Tr = Tr (M.Map String String)
  deriving (Semigroup, Monoid)

lookupTr :: String -> Tr -> Maybe String
lookupTr t (Tr m) = M.lookup t m

fromPairs :: [(String, String)] -> Tr
fromPairs = Tr . M.fromList

main :: IO ()
main = do
  putStrLn (maybe "none" id (lookupTr "abstract" mempty))
  let both = fromPairs [("abstract", "Summary")] <> fromPairs [("figure", "Fig.")]
  putStrLn (maybe "none" id (lookupTr "figure" both))
  putStrLn (maybe "none" id (lookupTr "appendix" (mconcat [mempty, both])))
