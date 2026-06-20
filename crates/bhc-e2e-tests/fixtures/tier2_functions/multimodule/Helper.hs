module Helper (double, Color(..), code) where

double :: Int -> Int
double x = x * 2

data Color = Red | Green | Blue

code :: Color -> Int
code Red = 1
code Green = 2
code Blue = 3
