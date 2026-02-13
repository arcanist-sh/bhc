module Main where

data Config = Config { width :: Int, height :: Int, title :: String }

showConfig :: Config -> String
showConfig c = title c ++ ": " ++ show (width c) ++ "x" ++ show (height c)

main :: IO ()
main = do
  let cfg = Config { width = 800, height = 600, title = "Window" }
  putStrLn (showConfig cfg)
  let cfg2 = cfg { width = 1024, height = 768 }
  putStrLn (showConfig cfg2)
  let cfg3 = cfg2 { title = "Resized" }
  putStrLn (showConfig cfg3)
