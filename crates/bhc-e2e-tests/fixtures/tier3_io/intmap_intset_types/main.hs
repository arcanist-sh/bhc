import qualified Data.IntMap as IntMap
import qualified Data.IntSet as IntSet

main :: IO ()
main = do
  let m = IntMap.fromList [(3, "c"), (1, "a"), (2, "b")]
  print (IntMap.lookup 1 (IntMap.fromList [(1, 0 :: Int)]))
  print (IntMap.lookup 9 m)
  print m
  print (IntMap.keys m)
  print (IntMap.elems m)
  print (IntMap.toList (IntMap.insertWith (++) 1 "z" m))
  print (IntMap.findWithDefault "?" 7 m)
  print (IntMap.size (IntMap.delete 2 m), IntMap.member 3 m)
  print (IntMap.map length m)
  print (IntMap.foldr (\v acc -> length v + acc) 0 m)
  print (IntMap.unionWith (++) m (IntMap.singleton 1 "x"))
  let s = IntSet.fromList [5, 1, 3, 1]
  print s
  print (IntSet.toList (IntSet.insert 2 s), IntSet.member 3 s, IntSet.size s)
