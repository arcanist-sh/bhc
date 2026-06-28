-- `print` of a Rational (idiomatic `import Data.Ratio`) used to print the
-- heap pointer on native, while `show` worked.
import Data.Ratio ((%))

main :: IO ()
main = do
    print (1 % 2 + 1 % 2)
    print (2 % 4)
    print (1 % 3 - 1 % 6)
