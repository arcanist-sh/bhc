-- Test Data.Map update, alter, unions
import qualified Data.Map as Map

main :: IO ()
main = do
    let m = Map.fromList [(1, 10), (2, 20), (3, 30)]
    -- update: modify existing key (Just)
    let m2 = Map.update (\v -> Just (v + 1)) 2 m
    putStrLn (show (Map.findWithDefault 0 2 m2))
    -- update: delete via Nothing
    let m3 = Map.update (\_ -> Nothing) 2 m
    putStrLn (show (Map.size m3))
    -- update: key not found
    let m4 = Map.update (\v -> Just (v * 2)) 9 m
    putStrLn (show (Map.size m4))
    -- alter: insert new key
    let m5 = Map.alter (\_ -> Just 99) 5 m
    putStrLn (show (Map.findWithDefault 0 5 m5))
    -- alter: delete existing
    let m6 = Map.alter (\_ -> Nothing) 1 m
    putStrLn (show (Map.size m6))
    -- alter: modify existing
    let m7 = Map.alter (\_ -> Just 130) 3 m
    putStrLn (show (Map.findWithDefault 0 3 m7))
    -- unions
    let ma = Map.fromList [(1, 100)]
    let mb = Map.fromList [(2, 200)]
    let mc = Map.fromList [(3, 300)]
    let mu = Map.unions [ma, mb, mc]
    putStrLn (show (Map.size mu))
    putStrLn "done"
