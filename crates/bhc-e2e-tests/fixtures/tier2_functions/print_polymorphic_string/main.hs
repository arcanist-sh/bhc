-- `print` of a `String` that a polymorphic function returns. Its expression
-- is typed `b`, and it printed as an address (#86).
pick :: Bool -> b -> b -> b
pick c x y = if c then x else y

applyTwice :: (b -> b) -> b -> b
applyTwice f x = f (f x)

firstOf :: [b] -> b
firstOf (x : _) = x
firstOf [] = error "empty"

main :: IO ()
main = do
  print (pick True "yes" "no")
  print (pick False "yes" "no")
  print (applyTwice ('a' :) "b")
  print (firstOf ["tab\there", "quote\"d"])
  print (applyTwice (++ "!") "")
  print (pick True (5 :: Int) 6)
  print (applyTwice reverse [1, 2, 3 :: Int])
  print "plain"
  print 'c'
  print ["a", "b"]
  print (Just "x")
