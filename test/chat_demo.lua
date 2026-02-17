-- Demo chat: progress callback, top-k probs, /embed command, kv_overrides.
-- Usage: luajit test/chat_demo.lua [model_path]
--        luajit test/chat_demo.lua --show-probs [model_path]
--        luajit test/chat_demo.lua --context-length 8192 [model_path]
--
-- Commands:
--   /embed <text>  -- run model:embed(text), print vector length + first 5 dims (embedding models only)
--   /probs         -- toggle showing top-5 token probs after each reply
--   (empty line)   -- quit

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

local args = {}
if arg then
	for i = 1, #arg do args[i] = arg[i] end
end

local show_probs = false
local context_length_override = nil
local model_path = nil
local i = 1
while i <= #args do
	local a = args[i]
	if a == "--show-probs" then
		show_probs = true
		i = i + 1
	elseif a == "--context-length" and args[i + 1] then
		context_length_override = tonumber(args[i + 1])
		i = i + 2
	elseif a:sub(1, 1) ~= "-" then
		model_path = a
		i = i + 1
	else
		i = i + 1
	end
end

model_path = model_path or "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"
local max_tokens = 1024

print("Loading model (progress below)...")
local backend = lluama.Backend()
backend:init()

local session_opts = {
	n_ctx = context_length_override or 2048,
	n_batch = 512,
	system_prompt = "You are a helpful assistant. Answer concisely.",
	temp = 0.8,
	progress_callback = function(p)
		io.write(string.format("\r  %.0f%% ", p * 100))
		io.flush()
	end,
}
if context_length_override then
	session_opts.kv_overrides = {
		{ key = "context_length", type = "int", value = context_length_override },
	}
end

local session = lluama.ChatSession(backend, model_path, session_opts)
print("\nReady. Commands: /embed <text> | /probs | empty to quit.")
print()

while true do
	io.write("You: ")
	io.flush()
	local line = io.read("*l")
	if not line then break end
	line = line:match("^%s*(.-)%s*$") or line
	if line == "" then break end

	-- /embed <text>
	if line:sub(1, 7) == "/embed " then
		local text = line:sub(8):match("^%s*(.-)%s*$") or line:sub(8)
		if text == "" then
			print("  Usage: /embed <text>")
		else
			local ok, vec = pcall(function() return session.model:embed(text, { pooling_type = "mean" }) end)
			if not ok then
				print("  embed failed (model may not support encoding): " .. tostring(vec))
			elseif not vec or #vec == 0 then
				print("  Empty embedding (encoder/embedding models only).")
			else
				print("  Vector length: " .. #vec)
				local head = ""
				for j = 1, math.min(5, #vec) do
					head = head .. string.format("%.4f ", vec[j])
				end
				print("  First 5 dims: " .. head)
			end
		end
		goto continue
	end

	-- /probs toggle
	if line == "/probs" then
		show_probs = not show_probs
		print("  Show top-5 probs after reply: " .. (show_probs and "on" or "off"))
		goto continue
	end

	-- Normal chat
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

		-- Top-k probs for last sampled token (index 0 = most recent)
		if show_probs then
			local top = session.ctx:sampled_top_k_ith(0, 5)
			if top and #top > 0 then
				io.write("  [top-5 last token]: ")
				for _, x in ipairs(top) do
					local piece = session.model:token_to_piece(x.id) or ("<id=" .. tostring(x.id) .. ">")
					piece = piece:gsub("\n", "\\n")
					io.write(string.format(" %s=%.3f", piece, x.p))
				end
				print()
			end
		end
	end

	::continue::
end
