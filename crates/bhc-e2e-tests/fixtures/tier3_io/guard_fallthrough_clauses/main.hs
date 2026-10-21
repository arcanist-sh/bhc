-- A clause whose guards all fail resumes at the next clause, for
-- multi-argument functions and for local functions in `where`, `let … in`
-- and `do`-block `let` alike; a `do`-block `let` keeps all of its clauses.
module Main where

top :: Int -> String -> String
top _ "x" = "x"
top n s | n == 1 = "one " ++ s
top _ s = "rest " ++ s

pick :: Int -> Int -> String
pick a b | a > b = "gt"
pick a b | a == b = "eq"
pick _ _ = "lt"

lk :: Int -> [(Int, String)] -> Maybe String
lk _ [] = Nothing
lk k ((k', v) : rest)
  | k == k' = Just v
  | otherwise = lk k rest

viaWhere :: Int -> String
viaWhere n = k n "z" ++ k 9 "z" ++ k 0 "q"
  where
    k _ "q" = "Q"
    k a s | a > 5 = "big"
    k _ s = s

viaLet :: Int -> String
viaLet n =
  let k _ "q" = "Q"
      k a s | a > 5 = "big"
      k _ s = s
   in k n "z" ++ k 9 "z" ++ k 0 "q"

-- pandoc's strToHtml helper: a guarded one-character clause between
-- literal clauses and a catch-all.
render :: [String] -> String
render = foldl go ""
  where
    go h "'" = h ++ "<apos>"
    go h "\"" = h ++ "<quot>"
    go h txt | length txt == 1 && all (== '~') txt = h ++ "<tilde>"
    go h txt = h ++ txt

main :: IO ()
main = do
  putStrLn (top 2 "y")
  putStrLn (top 1 "y")
  putStrLn (top 1 "x")
  putStrLn (pick 1 2 ++ pick 2 2 ++ pick 3 2)
  print (lk 2 [(1, "a"), (2, "b")])
  print (lk 3 [(1, "a"), (2, "b")])
  putStrLn (viaWhere 1)
  putStrLn (viaLet 1)
  let k _ "q" = "Q"
      k a s | a > 5 = "big"
      k _ s = s
  putStrLn (k (1 :: Int) "z" ++ k 9 "z" ++ k 0 "q")
  putStrLn (render ["a", "\"", "b", "~", "'", "cd"])
