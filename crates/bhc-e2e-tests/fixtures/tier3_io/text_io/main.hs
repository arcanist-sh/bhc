import Data.Text.IO

main :: IO ()
main = do
    let msg = Data.Text.pack "Hello from Text.IO"
    Data.Text.IO.writeFile "test_output.txt" msg
    contents <- Data.Text.IO.readFile "test_output.txt"
    Data.Text.IO.putStrLn contents
    Data.Text.IO.appendFile "test_output.txt" (Data.Text.pack "\nAppended line")
    contents2 <- Data.Text.IO.readFile "test_output.txt"
    Data.Text.IO.putStrLn contents2
    Data.Text.IO.putStr (Data.Text.pack "no newline")
    Data.Text.IO.putStrLn (Data.Text.pack "")
