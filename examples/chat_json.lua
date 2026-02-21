-- CLI chat with JSON grammar: forces model output to be valid JSON (object or array).
-- Run from repo root:
--   luajit examples/chat_json.lua [model_path]
--   luajit examples/chat_json.lua --no-grammar [model_path]  -- run without grammar (no constraint)

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

local no_grammar = (arg[1] == "--no-grammar")
if no_grammar then table.remove(arg, 1) end

local model_path = arg[1] or "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"
local max_tokens = 1024

local backend = lluama.Backend()
backend:init()

local session_opts = {
	template = "qwen",
	system_prompt = "You are a helpful assistant.",
	-- system_prompt = "You are a helpful assistant. Respond only with valid JSON. No markdown, no explanation, only a single JSON value (object or array).",
	temp = 0.7,
}
if not no_grammar then session_opts.grammar = "json" end

local session = lluama.ChatSession(backend, model_path, session_opts)

print("Ready (JSON mode). Type a message and press Enter (empty to quit).")
while true do
	io.write("You: ")
	io.flush()
	local line = io.read("*l")
	if not line then break end
	line = line:match("^%s*(.-)%s*$") or line
	if line == "" then break end

	local err = session:prompt(line)
	if err ~= 0 then
		io.stderr:write("Decode error: " .. tostring(err) .. "\n")
	else
		io.write("Model: ")
		io.flush()
		session:generate(max_tokens, function(piece)
			io.write(piece)
			io.flush()
		end)
		print()
	end
end
