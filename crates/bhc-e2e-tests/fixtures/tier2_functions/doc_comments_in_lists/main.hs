-- Haddock comments inside export and import lists (between items, after an
-- item before the comma, and as `-- *` section headers). These used to make
-- the list parser stop early and report a spurious "expected `)`".
module Main
    ( main
    -- * A section header in the export list
    -- | a doc comment in the export list
    , answer
    ) where

import Data.Char
    ( ord
    -- | a doc comment in the import list
    , chr
    )

answer :: Int
answer = ord 'A'

main :: IO ()
main = do
    print answer
    putStrLn [chr 66, chr 67]
