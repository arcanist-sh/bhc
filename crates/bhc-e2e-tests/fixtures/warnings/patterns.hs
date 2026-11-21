-- An incomplete and a redundant pattern match: both compile, with warnings
-- that point at the definition and at the equation never reached.
isTrue :: Bool -> Int
isTrue True = 1

classify :: Int -> String
classify _ = "any"
classify 0 = "zero"

main :: IO ()
main = putStrLn (show (isTrue True) ++ classify 1)
