-- Data.List sort/nub/intercalate and Data.Maybe.mapMaybe. These exercise the
-- WASM backend's synthesized prelude (build_list_fn) and the print/show
-- dispatch for runtime lists, both of which previously failed on WASM.
import Data.List (sort, nub, intercalate)
import Data.Maybe (mapMaybe)

main :: IO ()
main = do
    print (sort [3, 1, 4, 1, 5, 9, 2, 6 :: Int])
    print (nub [1, 1, 2, 3, 3, 3, 4, 1 :: Int])
    print (sort (nub [5, 3, 5, 1, 3, 1 :: Int]))
    print (mapMaybe (\x -> if even x then Just (x * x) else Nothing) [1 .. 8 :: Int])
    putStrLn (intercalate ", " ["alpha", "beta", "gamma"])
    putStrLn (intercalate "-" ["1", "2", "3"])
