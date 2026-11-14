-- A Map/Set built from String keys computed at run time (`words`, a list
-- comprehension) merges equal keys; codegen could not see the key type of a
-- comprehension and compared the keys by address, keeping every duplicate.
import qualified Data.Map as M
import qualified Data.Set as S

main :: IO ()
main = do
  let ws = words "a b a c b a"
  print (M.toList (M.fromListWith (+) [(w, 1 :: Int) | w <- ws]))
  print (M.toList (M.fromList [(w, length w) | w <- ws]))
  print (S.toList (S.fromList ws))
  print (M.lookup "a" (M.fromListWith (+) [(w, 1 :: Int) | w <- ws]))
