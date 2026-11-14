-- Data.Map's key-aware traversals at their real types; typed as `a -> b`,
-- `M.foldrWithKey (\k v acc -> k ++ show v ++ acc) "" m` printed an address.
import qualified Data.Map as M

main :: IO ()
main = do
  let m = M.fromList [("aa", 2 :: Int), ("b", 1), ("ccc", 3)]
  print (M.foldrWithKey (\k v acc -> k ++ show v ++ acc) "" m)
  print (M.foldlWithKey (\acc k v -> acc + length k + v) 0 m)
  print (M.toList (M.mapWithKey (\k v -> k ++ show v) m))
  print (M.toList (M.filterWithKey (\k v -> length k > 1 || v > 5) m))
  print (M.keys m, M.elems m)
