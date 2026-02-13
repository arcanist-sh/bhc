module Main where

data Color = Color { red :: Int, green :: Int, blue :: Int }

data Rect = Rect { rwidth :: Int, rheight :: Int, color :: Color }

showColor :: Color -> String
showColor c = "(" ++ show (red c) ++ "," ++ show (green c) ++ "," ++ show (blue c) ++ ")"

area :: Rect -> Int
area r = rwidth r * rheight r

main :: IO ()
main = do
  let c = Color { red = 255, green = 128, blue = 0 }
  let r = Rect { rwidth = 10, rheight = 20, color = c }
  putStrLn (showColor (color r))
  putStrLn (show (area r))
  let r2 = r { color = Color { red = 0, green = 255, blue = 0 } }
  putStrLn (showColor (color r2))
  putStrLn (show (area r2))
