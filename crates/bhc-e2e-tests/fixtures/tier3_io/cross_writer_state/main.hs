-- Cross-transformer test: WriterT over StateT
-- Stack: WriterT String (StateT Int IO) a
-- Uses: tell, lift get, lift put, runWriterT, runStateT

comp :: WriterT String (StateT Int IO) Int
comp = do
    s <- lift get
    tell "hello"
    lift (put (s + 10))
    tell " world"
    return s

main :: IO ()
main = do
    outerPair <- runStateT (runWriterT comp) 5
    let innerPair = fst outerPair
    let finalState = snd outerPair
    let result = fst innerPair
    let log = snd innerPair
    putStrLn (show result)
    putStrLn log
    putStrLn (show finalState)
