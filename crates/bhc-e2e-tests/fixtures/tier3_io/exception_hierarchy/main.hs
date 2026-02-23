-- Exception hierarchy test
-- Demonstrates: SomeException wrapping, error catchability, IO error catching

import Control.Exception

main :: IO ()
main = do
    -- Test 1: catch-all still works (backward compat)
    catch
        (throwIO "test error")
        (\e -> putStrLn "caught all")

    -- Test 2: error is now catchable
    catch
        (error "boom")
        (\e -> putStrLn "caught error")

    -- Test 3: IO error from readFile is catchable
    catch
        (do contents <- readFile "nonexistent_file_xyz.txt"
            putStrLn contents)
        (\e -> putStrLn "caught IO error")

    -- Test 4: done
    putStrLn "done"
