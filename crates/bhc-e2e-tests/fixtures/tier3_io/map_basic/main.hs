-- Test basic Data.Map operations
import qualified Data.Map as Map

main :: IO ()
main = do
    let m1 = Map.fromList [(1, 10), (2, 20), (3, 30)]
    putStrLn (show (Map.size m1))
    putStrLn (show (Map.member 2 m1))
    putStrLn (show (Map.member 5 m1))
    let m2 = Map.insert 4 40 m1
    putStrLn (show (Map.size m2))
    let m3 = Map.delete 2 m2
    putStrLn (show (Map.size m3))
    putStrLn (show (Map.null m1))
    let v = Map.findWithDefault 0 1 m1
    putStrLn (show v)
    let v2 = Map.findWithDefault 0 9 m1
    putStrLn (show v2)
