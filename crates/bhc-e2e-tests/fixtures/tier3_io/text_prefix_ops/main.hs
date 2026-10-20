-- Test: Data.Text prefix operations driven by a predicate
-- Verifies: span/break/takeWhile/dropWhile/dropWhileEnd/dropAround split at
-- the right character (span, takeWhile and dropWhile used to stop at the
-- first character SATISFYING the predicate).

import qualified Data.Text as T
import qualified Data.Text.IO as TIO

-- A multi-clause predicate: its Bool comes back as a constructor, not 0/1.
isWS :: Char -> Bool
isWS ' ' = True
isWS '\n' = True
isWS _ = False

bar :: (T.Text, T.Text) -> T.Text
bar (a, b) = T.concat [a, T.pack "|", b]

main :: IO ()
main = do
  let t = T.pack "aab"
  TIO.putStrLn (bar (T.span (== 'a') t))
  TIO.putStrLn (bar (T.break (== 'b') t))
  TIO.putStrLn (T.takeWhile (== 'a') t)
  TIO.putStrLn (T.dropWhile (== 'a') t)
  TIO.putStrLn (T.dropWhileEnd (== 'b') t)
  TIO.putStrLn (T.dropAround (== '`') (T.pack "``code``"))
  let trim = T.dropAround (== ' ')
  TIO.putStrLn (T.append (trim (T.pack "  x y  ")) (T.pack "|"))
  TIO.putStrLn (T.append (T.dropWhileEnd isWS (T.pack "  code \n")) (T.pack "|"))
  TIO.putStrLn (T.append (T.takeWhileEnd isWS (T.pack "ab  ")) (T.pack "|"))
