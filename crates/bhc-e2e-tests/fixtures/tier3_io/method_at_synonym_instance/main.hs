{-# LANGUAGE FlexibleInstances #-}
-- A class method resolved at a type written through a synonym. The instances
-- are declared at `MkM ()` and `MkM () -> MkM ()`; the uses are typed with
-- `Html`. Unexpanded, `Html -> Html` named no instance, and `section ! a ! b
-- $ x` took the instance for markup and called the result as a function.
data MkM a = Leaf String | Append (MkM a) (MkM a) | Attr String (MkM a)

type Markup = MkM ()
type Html = Markup

newtype Attribute = Attribute (Html -> Html)

class Attributable h where
  (!) :: h -> Attribute -> h

instance Attributable (MkM ()) where
  (!) h (Attribute f) = f h

instance Attributable (MkM () -> MkM ()) where
  (!) g (Attribute f) = \x -> f (g x)

infixl 8 !

attr :: String -> Attribute
attr s = Attribute (Attr s)

section :: Html -> Html
section x = Append (Leaf "<s>") (Append x (Leaf "</s>"))

render :: Html -> String
render (Leaf s) = s
render (Append x y) = render x ++ render y
render (Attr a m) = "[" ++ a ++ "]" ++ render m

build :: Bool -> Int -> [Html] -> Html
build html5 n notes =
  let container x
        | html5
        , n == 3
                = section ! attr "a"
                          ! attr "b"
                          ! attr "c"
                          $ x
        | html5 = section ! attr "d" ! attr "e" $ x
        | otherwise = section ! attr "f" $ x
  in if null notes
       then Leaf ""
       else container (foldr Append (Leaf "") notes)

main :: IO ()
main = do
  putStrLn (render (build True 3 [Leaf "x"]))
  putStrLn (render (build True 1 [Leaf "x", Leaf "y"]))
  putStrLn (render (build False 1 [Leaf "x"]))
  putStrLn (render (section (Leaf "v") ! attr "g" ! attr "h"))
