-- Built-in GBNF grammars for constrained decoding.
-- JSON grammar uses a prefix so ChatSession can accept the chat prompt (prefix) then generate object only.
-- grammar_root for JSON is "root".

local json_gbnf = [[
root ::= prefix value
prefix ::= .* "\n"
value ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
 "{" ws (
 string ":" ws value
 ("," ws string ":" ws value)*
 )? "}" ws

array ::=
 "[" ws (
 value
 ("," ws value)*
 )? "]" ws

string ::=
 "\"" (
 [^"\\\x7F\x00-\x1F] |
 "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
 )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= [ \t\n\r]*
]]

return {
	json = json_gbnf,
	json_root = "root",
}
