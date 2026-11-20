-- | Show-constrained functions, called from another module.
module ShowLib (label, labelled, bracketed) where

import qualified Data.Text as T

label :: Show a => Maybe Int -> a -> String
label _ x = show x

labelled :: Show a => a -> T.Text
labelled x = T.pack ("<" ++ show x ++ ">")

bracketed :: Show a => [a] -> String
bracketed xs = concatMap (\x -> "[" ++ show x ++ "]") xs
