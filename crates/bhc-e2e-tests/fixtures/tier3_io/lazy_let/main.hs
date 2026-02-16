module Main where

-- Test that unused error bindings don't crash (lazy let-bindings)

main :: IO ()
main = do
    -- Unused error binding should not be evaluated
    let x = error "this should not be evaluated"
    putStrLn "before"
    let y = 42 :: Int
    print y
    putStrLn "after"
    -- x is never used, so program should succeed
