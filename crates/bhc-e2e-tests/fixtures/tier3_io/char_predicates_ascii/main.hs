module Main where

import Data.Char (isAsciiLower, isAsciiUpper, isLatin1, isOctDigit, isSeparator)

-- pandoc's `anyOrderedListMarker` aborted on `stub: isAsciiLower`.
describe :: Char -> String
describe c =
  [ pick (isAsciiLower c) 'l'
  , pick (isAsciiUpper c) 'u'
  , pick (isOctDigit c) 'o'
  , pick (isSeparator c) 's'
  , pick (isLatin1 c) '1'
  ]
  where
    pick b ch = if b then ch else '.'

main :: IO ()
main = mapM_ (putStrLn . describe) "aZ7 \233"
