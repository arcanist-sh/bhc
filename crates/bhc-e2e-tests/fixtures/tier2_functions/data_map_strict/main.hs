-- `Data.Map.Strict` and `Data.Map.Lazy` reach the same container primops as
-- `Data.Map`; they ran as "not implemented" stubs (issue #26).
import qualified Data.Map.Strict as M
import qualified Data.Map.Lazy as L

main :: IO ()
main = do
  print (M.toList (M.fromList [(1::Int,'a')]))
  let m = M.insert 2 'b' (M.fromList [(1::Int,'a')])
  print (M.size m)
  print (M.lookup 2 m)
  print (L.toList (L.fromList [(3::Int,"c")]))
