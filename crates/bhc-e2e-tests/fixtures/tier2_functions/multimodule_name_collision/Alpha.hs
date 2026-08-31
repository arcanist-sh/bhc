module Alpha (tag) where

-- Same name as Beta.tag, different arity. Whichever module is imported
-- first claims the bare name; the other must still reach its own symbol.
tag :: Int -> Int
tag n = n + 1
