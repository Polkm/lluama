-- Demo chat: optional progress bar, top-k probs, /embed command, kv_overrides.
-- Usage: luajit test/chat_demo.lua [model_path]
--        luajit test/chat_demo.lua --show-probs [model_path]
--        luajit test/chat_demo.lua --progress [model_path]
--        luajit test/chat_demo.lua --context-length 8192 [model_path]
--
-- Commands: /embed <text> | /probs | empty to quit

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

local args = {}
if arg then for i = 1, #arg do args[i] = arg[i] end end

local show_probs = false
local want_progress = false
local context_length_override = nil
local model_path = nil
local i = 1
while i <= #args do
	local a = args[i]
	if a == "--show-probs" then
		show_probs = true
		i = i + 1
	elseif a == "--progress" then
		want_progress = true
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

local backend = lluama.Backend()
backend:init()

-- Match chat.lua: only add options that don't change load behavior unless requested.
local session_opts = {
	n_ctx = context_length_override or 2048,
	n_batch = 512,
	system_prompt = "You are a helpful assistant. Answer concisely.",
	temp = 0.8,
}
if want_progress then
	session_opts.progress_callback = function(p)
		io.write(string.format("\r  %.0f%% ", p * 100))
		io.flush()
	end
end
if context_length_override then
	session_opts.kv_overrides = {
		{ key = "context_length", type = "int", value = context_length_override },
	}
end

if want_progress then print("Loading model (progress below)...") end
local session = lluama.ChatSession(backend, model_path, session_opts)
if want_progress then print("\n") end
print("Ready. Commands: /embed <text> | /probs | empty to quit.")
print()

while true do
	io.write("You: ")
	io.flush()
	local line = io.read("*l")
	if not line then break end
	line = line:match("^%s*(.-)%s*$") or line
	if line == "" then break end

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
				for j = 1, math.min(5, #vec) do head = head .. string.format("%.4f ", vec[j]) end
				print("  First 5 dims: " .. head)
			end
		end
		goto continue
	end

	if line == "/probs" then
		show_probs = not show_probs
		print("  Show top-5 probs after reply: " .. (show_probs and "on" or "off"))
		goto continue
	end

	local err = session:prompt(line)
	if err ~= 0 then
		io.stderr:write("Decode error: " .. tostring(err) .. "\n")
	else
		local last_top_k = nil
		if show_probs then
			session._on_before_sample = function(ctx, logits_idx)
				last_top_k = ctx:logits_top_k_ith(logits_idx, 5)
			end
		else
			session._on_before_sample = nil
		end
		io.write("Model: ")
		io.flush()
		session:generate(max_tokens, function(piece)
			io.write(piece)
			io.flush()
		end)
		print()
		if show_probs and last_top_k and #last_top_k > 0 then
			io.write("  [top-5 last token]: ")
			for _, x in ipairs(last_top_k) do
				local piece = session.model:token_to_piece(x.id) or ("<id=" .. tostring(x.id) .. ">")
				piece = piece:gsub("\n", "\\n")
				io.write(string.format(" %s=%.3f", piece, x.p))
			end
			print()
		end
	end
	::continue::
end
