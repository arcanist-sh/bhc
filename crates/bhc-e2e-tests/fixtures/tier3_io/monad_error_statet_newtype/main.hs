{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- throwError/catchError at a StateT over a newtype that derives its
-- MonadError: the StateT instance must be used, not the newtype's own.
import Control.Monad.Except (MonadError, catchError, throwError, runExceptT, ExceptT)
import Control.Monad.State.Strict (StateT, evalStateT)

newtype P a = P { unP :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad, MonadError String)

runP :: P a -> IO (Either String a)
runP p = evalStateT (runExceptT (unP p)) 0

type S = StateT () P

caughtInP :: P String
caughtInP = catchError (throwError "boom" >> return "no") (\e -> return ("caught " ++ e))

thrownInS :: S String
thrownInS = throwError "boom" >> return "no"

caughtInS :: S String
caughtInS = catchError (throwError "boom" >> return "no") (\e -> return ("caught " ++ e))

out :: Either String String -> IO ()
out = either (\e -> putStrLn ("Left " ++ e)) putStrLn

main :: IO ()
main = do
  runP caughtInP >>= out
  runP (evalStateT thrownInS ()) >>= out
  runP (evalStateT caughtInS ()) >>= out
