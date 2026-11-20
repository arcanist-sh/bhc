-- An infix definition whose left operand is a constructor pattern, at top
-- level: `TagOpen n as ~== TagOpen p ps = …` (tagsoup's `~==`, which pandoc
-- vendors). It was a parse error ("unexpected constructor, expected
-- declaration"); in a `where` it already parsed. Under the eager IO model the
-- `n == p` on two pattern-bound Strings, in a function with a guarded clause,
-- compared addresses.
data Tag = TagOpen String [String] | TagText String

data Box = Box Int

(~==) :: Tag -> Tag -> Bool
TagOpen n atts ~== TagOpen p patts = n == p && present
  where
    present = length atts >= length patts
TagText a ~== TagText b
  | null b = True
  | otherwise = a == b
_ ~== _ = False

plus :: Box -> Box -> Box
Box a `plus` Box b = Box (a + b)

unbox :: Box -> Int
unbox (Box n) = n

Box seven = Box 7

main :: IO ()
main = do
  print (TagOpen "a" ["x"] ~== TagOpen "a" [])
  print (TagOpen "a" [] ~== TagOpen "a" ["x"])
  print (TagText "t" ~== TagText "")
  print (TagText "t" ~== TagText "u")
  print (TagText "t" ~== TagOpen "t" [])
  print (unbox (Box 2 `plus` Box 3))
  print seven
  print (unbox (Box 1 <+> Box 4))
  where
    Box a <+> Box b = Box (a * 10 + b)
