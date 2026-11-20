-- Under OverloadedStrings a string literal used as a list is a list of Char:
-- the only list instance of IsString is (a ~ Char) => IsString [a]. The
-- element type stayed unknown when nothing else fixed it, and the
-- characters of `zip [1 ..] "ab"` printed as numbers.
{-# LANGUAGE OverloadedStrings #-}
import qualified Data.Text as T

main :: IO ()
main = do
  print (zip [1 :: Int ..] "ab")
  print (map fst (zip "xy" [True, False]), length "abc", reverse "abc")
  print (T.length "abc", T.unpack "t", "s" ++ "u", lookup 'b' (zip "ab" [1 :: Int ..]))
  let t = "text" :: T.Text
  print (t, T.append t "!", [c | c <- "pq"], words "a b")
  mapM_ putStrLn ["one", "two"]
