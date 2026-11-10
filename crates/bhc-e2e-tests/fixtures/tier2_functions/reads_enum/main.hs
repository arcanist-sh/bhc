-- `reads` at an enumeration with derived Read parses a constructor name
-- (after optional whitespace) and returns the rest of the input; it read an
-- Int and returned [] for every constructor. Pandoc's `readExtension` is
-- `case reads ("Ext_" ++ name) of [(x, "")] -> Just x; _ -> Nothing`.
data Color = Red | Green | Blue deriving (Show, Read, Eq)

data Extension = Ext_raw | Ext_raw_html | Ext_smart | Ext_raw_tex
  deriving (Show, Read, Eq)

readExtension :: String -> Maybe Extension
readExtension name = case reads ("Ext_" ++ name) of
  [(x, "")] -> Just x
  _ -> Nothing

main :: IO ()
main = do
  print (reads "Green" :: [(Color, String)])
  print (reads "Green rest" :: [(Color, String)])
  print (reads "  Blue" :: [(Color, String)])
  print (reads "Purple" :: [(Color, String)])
  print (reads "Greenish" :: [(Color, String)])
  print (reads "42 x" :: [(Int, String)])
  print (map readExtension ["raw", "raw_html", "smart", "raw_tex", "nope", "raw "])
