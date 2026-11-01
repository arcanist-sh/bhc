-- concatMap (and concat) over a list that arrives unevaluated: a let-bound
-- helper's parameter bound to a lazy recursive call.
perms :: [Int] -> [[Int]]
perms [] = [[]]
perms (y:ys) =
  let bnd m k = concatMap k m
  in bnd [y, y * 2] (\r -> bnd (perms ys) (\rs -> [r : rs]))

flat :: [[Int]] -> [Int]
flat xss = let c ys = concat ys in c xss

main :: IO ()
main = do
  print (perms [1, 10])
  print (flat (perms [3]))
