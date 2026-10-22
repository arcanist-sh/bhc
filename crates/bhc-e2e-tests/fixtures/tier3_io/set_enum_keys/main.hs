-- Sets and maps keyed by an enumeration compare keys by constructor, not by
-- address, also when the type has a constructor with a field after its
-- nullary ones (pandoc's `Extension` with `CustomExtension Text`).
{-# LANGUAGE DeriveGeneric #-}
module Main where

import GHC.Generics (Generic)

import qualified Data.Map as M
import qualified Data.Set as Set

data Ext = ExtA | ExtB | ExtC | ExtD | Custom String
  deriving (Eq, Ord, Show)

-- A derived Generic instance must not make this look like a non-enum.
data Flag = FlagX | FlagY | FlagZ | Other String
  deriving (Eq, Ord, Show, Generic)

newtype Exts = Exts (Set.Set Ext)

enabled :: Ext -> Exts -> Bool
enabled x (Exts s) = Set.member x s

defaults :: Exts
defaults = Exts (Set.fromList [ExtA, ExtC])

main :: IO ()
main = do
  print (enabled ExtA defaults)
  print (enabled ExtB defaults)
  print (enabled ExtC defaults)
  print (Set.size (Set.fromList [ExtA, ExtB, ExtA, ExtB]))
  print (Set.member ExtD (Set.insert ExtD Set.empty))
  print (Set.member ExtB (Set.delete ExtB (Set.fromList [ExtA, ExtB])))
  print (M.lookup ExtB (M.fromList [(ExtA, "a"), (ExtB, "b")]))
  print (flagOn FlagY (Set.fromList [FlagX, FlagY]))
  print (flagOn FlagZ (Set.fromList [FlagX, FlagY]))

flagOn :: Flag -> Set.Set Flag -> Bool
flagOn f s = f `Set.member` s
