main = do
  bracket (putStrLn "acquire") (\_ -> putStrLn "release") (\_ -> putStrLn "use")
  putStrLn "done"
