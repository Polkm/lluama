-- Basic chat with a simple grammar: each reply is constrained to "yes" or "no".
-- Uses the chat template so the model sees normal context. The prompt must end with a newline
-- so that prefix (.* "\n") is fully matched before we generate; then only "yes"|"no" is allowed.
-- Usage: luajit test/chat_grammar.lua [model_path]

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

local model_path = arg[1] or "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"
local max_tokens = 64

local templates = lluama.chat_templates
local t = templates.get("qwen")
assert(t, "qwen template required")

-- Grammar: prefix (any chars ending with newline) then "yes" or "no". Prompt must end with \n.
local gbnf = [[
root ::= prefix ("yes" | "no")
prefix ::= .* "\n"
]]

local llama = lluama.llama

local backend = lluama.Backend()
backend:init()

local model = lluama.Model(backend, model_path)
local ctx = model:context({ n_ctx = 512, n_batch = 256 })
local eos = llama.llama_vocab_eos(model:vocab())

local sampler = lluama.Sampler({
	grammar = gbnf,
	grammar_root = "root",
	temp = 0.5,
	seed = 42,
}, model)
ctx:set_sampler(sampler)

local state = { n_past = 0, first_turn = true }

local function generate_one_reply(user_message)
	sampler:reset()
	local prompt
	if state.first_turn then
		prompt = templates.format_conversation(
			{ { role = "user", content = user_message } },
			"Answer only with the word yes or no. No explanation.",
			t
		)
		state.first_turn = false
	else
		prompt = t.user_start .. user_message .. t.user_end .. t.assistant_start
	end
	-- Ensure prompt ends with newline so grammar prefix is complete before we generate.
	if prompt:sub(-1) ~= "\n" then prompt = prompt .. "\n" end

	local tokens = model:tokenize(prompt, false)
	if #tokens == 0 then return "" end

	local err = ctx:decode_tokens(tokens, state.n_past)
	if err ~= 0 then return "" end

	for i = 1, #tokens do
		sampler:accept(tokens[i])
	end
	local n_past = state.n_past + #tokens

	local reply = ""
	local logits_idx = #tokens - 1
	for _ = 1, max_tokens do
		local next_token = sampler:sample(ctx.ctx, logits_idx)
		if next_token == eos then break end

		local piece = model:token_to_piece(next_token)
		reply = reply .. piece
		sampler:accept(next_token)
		err = ctx:decode_one(next_token, n_past)
		if err ~= 0 then break end
		n_past = n_past + 1
		logits_idx = 0

		local normalized = reply:gsub("%s+", ""):lower()
		if normalized:match("^yes") or normalized:match("^no") then break end
		if #reply > 20 then break end
	end

	state.n_past = n_past
	return reply
end

print("Grammar chat: each reply is constrained to 'yes' or 'no'. Type a message and press Enter (empty to quit).")
while true do
	io.write("You: ")
	io.flush()
	local line = io.read("*l")
	if not line then break end
	line = line:match("^%s*(.-)%s*$") or line
	if line == "" then break end

	io.write("Model: ")
	io.flush()
	local reply = generate_one_reply(line)
	reply = reply:gsub("^%s+", ""):gsub("%s+$", ""):lower()
	io.write(#reply > 0 and reply or "(no output)")
	io.write("\n")
	io.flush()
end
