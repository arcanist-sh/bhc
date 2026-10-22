-- `intersperse` and `intercalate` imported from Data.List keep their types,
-- so a Monoid method on their result finds its instance. Typed as fresh
-- variables, pandoc's `bodies <- intersperse nl <$> mapM …; mconcat bodies`
-- lost the element type and `mconcat` ran as a stub.
module Main where
import Data.List (intercalate, intersperse)

newtype Path = Path [String]

instance Semigroup Path where
  Path a <> Path b = Path (a ++ b)

instance Monoid Path where
  mempty = Path []

unPath :: Path -> [String]
unPath (Path xs) = xs

sep :: Path
sep = Path ["/"]

joined :: Maybe [Path] -> Maybe Path
joined mps = do
  parts <- intersperse sep <$> mps
  return (mconcat parts)

main :: IO ()
main = do
  case joined (Just [Path ["a"], Path ["b"], Path ["c"]]) of
    Just p -> putStrLn (concat (unPath p))
    Nothing -> putStrLn "none"
  putStrLn (intercalate ", " ["x", "y", "z"])
  print (length (intercalate [0] [[1, 2], [3 :: Int]]))
