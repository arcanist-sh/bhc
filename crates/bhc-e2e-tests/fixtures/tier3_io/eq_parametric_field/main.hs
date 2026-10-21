-- `==` on fields of a parametric constructor instantiated at Text, Char and
-- Int compares by value, in top-level, case and generalised local code
-- (tagsoup's `~==` and minimised-tag rendering); a `Char` predicate handed
-- to `T.break` still takes its raw character.
module Main where

import qualified Data.Text as T

data Tg s = Op s | Cl s

same :: Tg T.Text -> Tg T.Text -> Bool
same (Op a) (Op b) = T.null b || a == b
same (Cl a) (Cl b) = T.null b || a == b
same _ _ = False

sameChar :: Tg Char -> Tg Char -> Bool
sameChar x y = case (x, y) of
  (Cl a, Cl b) -> a == b
  _ -> False

differ :: Tg Int -> Tg Int -> Bool
differ (Op a) (Op b) = a /= b
differ _ _ = True

render :: [Tg T.Text] -> String
render = go False
  where
    go _ [] = ""
    go r (Op a : Cl b : rest)
      | a == b = "<" ++ T.unpack a ++ "/>" ++ go r rest
    go r (Op a : rest) = "<" ++ T.unpack a ++ ">" ++ go r rest
    go r (Cl a : rest) = "</" ++ T.unpack a ++ ">" ++ go r rest

yn :: Bool -> String
yn b = if b then "yes" else "no"

main :: IO ()
main = do
  putStrLn (yn (same (Cl (T.pack "div")) (Cl (T.pack "div"))))
  putStrLn (yn (same (Op (T.pack "div")) (Op (T.pack "span"))))
  putStrLn (yn (same (Op (T.pack "div")) (Op T.empty)))
  putStrLn (yn (sameChar (Cl 'x') (Cl 'x')))
  putStrLn (yn (differ (Op 1) (Op 2)))
  putStrLn (render [Op (T.pack "br"), Cl (T.pack "br"), Op (T.pack "p"), Cl (T.pack "q")])
  print (T.length (fst (T.break (\c -> c == '\n' || c == '|') (T.pack "ab|cd"))))
