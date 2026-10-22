-- A constructor imported unqualified keeps its own tag when a `qualified`
-- import brings a same-named constructor of another type (pandoc's HTML
-- writer: Block's `Table` next to `qualified AnnotatedTable as Ann`'s
-- `Table`). Registering the qualified one under the bare name gave Block's
-- `Table` tag 0, so a table matched `Plain` — or no alternative at all.
module Main where
import Doc
import qualified Ann

annArea :: Ann.Table -> Int
annArea (Ann.Table a b) = a * b

describe :: Block -> String
describe (Plain n) = "plain " ++ show n
describe (Para n) = "para " ++ show n
describe (Table a b) = "table " ++ show (annArea (Ann.toTable a b))

main :: IO ()
main = mapM_ (putStrLn . describe) [Plain 1, Table 2 3, Para 4]
