module Defs (Inl(..), Doc(..)) where
import Data.Text (Text)

data Inl = Str Text | Emph [Inl] | Space

data Doc = Doc { docTitle :: Text, docBody :: [Inl] }
