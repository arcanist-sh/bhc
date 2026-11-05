-- Handlers see an exception they can show (#72), and `bracket_` runs its parts in
-- order (#71).
import Control.Exception

main :: IO ()
main = do
  catch (throwIO (userError "a")) (\e -> putStrLn ("caught " ++ show (e :: IOException)))
  handle (\e -> putStrLn ("handled " ++ show (e :: IOException))) (throwIO (userError "b"))
  r <- catch (return 1) (\e -> const (return 2) (e :: IOException))
  print (r :: Int)
  (putStrLn "body" >> throwIO (userError "c")) `catch` (\e -> putStrLn (show (e :: SomeException)))
  x <- bracket (putStrLn "acq" >> return 1) (\_ -> putStrLn "rel") (\v -> return (v + 1))
  print (x :: Int)
  bracket_ (putStrLn "b1") (putStrLn "b3") (putStrLn "b2")
  (throwIO (userError "d") `onException` putStrLn "on-exc")
    `catch` (\e -> print (e :: IOException))
  s <- evaluate (2 + 3 :: Int)
  print s
