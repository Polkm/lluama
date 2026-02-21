-- CLI chat using the library's ChatSession. Run from repo root.
-- Usage: luajit examples/chat.lua [model_path]
-- Requires a GGUF model with a chat template (tokenizer.chat_template in metadata).
-- If your model has no template, pass opts.template_string with the Jinja template.

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

local model_path = arg[1] or "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"
local max_tokens = 1024

local backend = lluama.Backend()
backend:init()

local session = lluama.ChatSession(backend, model_path, {
	system_prompt = "You are a helpful assistant. Answer concisely.",
	temp = 0.8,
})

print("Ready. Type a message and press Enter (empty to quit).")
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
