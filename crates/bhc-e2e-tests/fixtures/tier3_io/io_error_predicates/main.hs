-- The System.IO.Error predicates say what kind of IOException was caught.
-- They were stubs: a program could catch the exception but asking what it
-- was aborted with "stub: isDoesNotExistError not implemented".
import System.IO
import System.IO.Error
import Control.Exception

classify :: IOException -> String
classify e
  | isDoesNotExistError e = "does not exist"
  | isPermissionError e = "permission"
  | isAlreadyExistsError e = "already exists"
  | isUserError e = "user error"
  | isEOFError e = "end of file"
  | otherwise = "other"

main :: IO ()
main = do
  r1 <- try (readFile "/nonexistent-bhc/a.txt") :: IO (Either IOException String)
  putStrLn (either classify (const "no error") r1)
  r2 <- try (ioError (userError "boom")) :: IO (Either IOException ())
  putStrLn (either classify (const "no error") r2)
  r3 <- try (openFile "/nonexistent-bhc/b.txt" ReadMode) :: IO (Either IOException Handle)
  case r3 of
    Left e ->
      print
        ( isDoesNotExistError e
        , isUserError e
        , isPermissionError e
        , isFullError e
        , isAlreadyInUseError e
        , isIllegalOperation e
        )
    Right _ -> putStrLn "no error"
  r4 <- tryIOError (readFile "/nonexistent-bhc/c.txt")
  putStrLn (either (\e -> if isDoesNotExistError e then "tryIOError: missing" else "tryIOError: other") (const "no error") r4)
  s <- readFile "/nonexistent-bhc/d.txt" `catchIOError` \e ->
    return (if isDoesNotExistError e && not (isUserError e) then "handled missing file" else "handled other")
  putStrLn s
