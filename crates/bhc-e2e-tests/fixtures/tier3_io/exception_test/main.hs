-- Milestone E.5: Exception Handling Test
-- Demonstrates: catch, bracket, throwIO, finally, onException

main :: IO ()
main = do
    -- Test 1: catch file not found error
    catch
        (do contents <- readFile "nonexistent_file_12345.txt"
            putStrLn contents)
        (\e -> putStrLn "caught file error")

    -- Test 2: bracket ensures cleanup runs
    bracket
        (putStrLn "resource acquired")
        (\_ -> putStrLn "resource released")
        (\_ -> putStrLn "using resource")

    -- Test 3: catch with explicit throwIO
    catch
        (throwIO "explicit error")
        (\e -> putStrLn "caught throwIO")

    -- Test 4: finally ensures cleanup
    finally
        (putStrLn "action succeeded")
        (putStrLn "finally cleanup")

    -- Test 5: onException runs handler only on error
    catch
        (onException
            (throwIO "will throw")
            (putStrLn "onException handler"))
        (\e -> putStrLn "outer catch")

    putStrLn "all tests passed"
