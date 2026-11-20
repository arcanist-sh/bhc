-- Text functions that take a list (concat, intercalate, unwords, unlines)
-- walk a list that may be built lazily: a map over a zipWith on an infinite
-- range yields its tail on demand. The text library read the unevaluated
-- tail as a cons cell and aborted (Pandoc numbering its sections).
{-# LANGUAGE OverloadedStrings #-}
import qualified Data.Text as T

tshow :: Show a => a -> T.Text
tshow = T.pack . show

f :: Int -> [Int] -> [T.Text]
f k xs = map tshow (zipWith (+) [k ..] xs)

main :: IO ()
main = do
  let ts = f 1 [1, 2, 3]
  print (T.concat ts, T.intercalate "." ts, T.unwords ts, T.unlines ts)
  print (T.length (head ts), T.append (head ts) (last ts), map T.unpack ts, ts == ["2", "4", "6"])
  print (T.isPrefixOf (head ts) "24", T.toUpper (T.concat (map (T.cons 'x') ts)), mconcat ts, head ts <> "!")
