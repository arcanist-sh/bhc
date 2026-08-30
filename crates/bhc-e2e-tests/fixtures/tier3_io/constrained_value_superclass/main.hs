-- A constrained VALUE takes its dictionaries at its use site, and only from
-- the enclosing binding's OWN constraints. parsec's `getState :: Monad m =>
-- ParsecT s u m u` used inside a `PandocMonad m =>` function found no `Monad`
-- there — `PandocMonad` has it as a superclass — so it was emitted with its
-- dictionary missing and its first value argument landed in that slot.
module Main where

data Box a = Box a

unBox :: Box a -> a
unBox (Box a) = a

class Small f where
  small :: a -> f a

class Small f => Big f where
  big :: f Int

instance Small Box where
  small = Box

instance Big Box where
  big = Box 99

-- A nullary constrained value: no argument will ever pin its dictionary.
unit :: Small f => f Int
unit = small 1

-- Its use site is constrained by the SUBCLASS only.
useIt :: Big f => f Int
useIt = unit

main :: IO ()
main = do
  print (unBox (unit :: Box Int))
  print (unBox (useIt :: Box Int))
  print (unBox (big :: Box Int))
