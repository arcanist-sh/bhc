-- A data type whose constructors carry Haddock doc comments, including one
-- before the `=` (documents the first constructor). This used to drop every
-- constructor (parsed as an empty data decl), leaving them all unbound.
data Shape
    -- | a circle with a radius
    = Circle Int
    -- | a square with a side
    | Square Int
    -- | nothing
    | Blank
    deriving (Eq, Show)

area :: Shape -> Int
area s = case s of
    Circle r -> 3 * r * r
    Square x -> x * x
    Blank    -> 0

main :: IO ()
main = do
    print (area (Circle 2))
    print (area (Square 3))
    print (area Blank)
    print (Circle 2 == Circle 2)
