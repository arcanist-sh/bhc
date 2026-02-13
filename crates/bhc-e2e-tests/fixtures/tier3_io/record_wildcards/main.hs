{-# LANGUAGE RecordWildCards #-}
module Main where

data Config = Config { host :: String, port :: Int, debug :: Int }

showConfig :: Config -> String
showConfig (Config {..}) = host ++ ":" ++ show port ++ " debug=" ++ show debug

makeConfig :: String -> Int -> Config
makeConfig host port = Config {..}
  where debug = 0

showPartial :: Config -> String
showPartial (Config { host = h, .. }) = h ++ " port=" ++ show port

main :: IO ()
main = do
  let cfg = makeConfig "localhost" 8080
  putStrLn (showConfig cfg)
  let cfg2 = Config { host = "example.com", port = 443, debug = 1 }
  putStrLn (showConfig cfg2)
  putStrLn (showPartial cfg)
