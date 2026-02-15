-- Integration tests: load Qwen model and exercise full pipeline. Run in a loop to catch flakes.
-- Usage: luajit test/integration_test.lua [model_path] [--loop N] [--only NAME]
--   model_path  default: models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf
--   --loop N    run full suite N times (0 = infinite until Ctrl+C)
--   --only NAME run only test NAME
-- Run from repo root. Requires Qwen GGUF model at model_path.
-- Important: set_sampler(ctx, sampler) must be called before decode_tokens so the backend associates logits with the sequence.

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

-- Parse args: [model_path] [--loop N] [--only NAME]
local model_path = "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"
local loop_count = 1
local only_name = nil
local i = 1
while i <= #arg do
	if arg[i] == "--loop" then
		i = i + 1
		loop_count = tonumber(arg[i]) or 1
		if loop_count == 0 then loop_count = 1/0 end -- "infinite"
	elseif arg[i] == "--only" then
		i = i + 1
		only_name = arg[i]
	else
		model_path = arg[i]
	end
	i = i + 1
end

local llama = lluama.llama
local t = lluama.chat_templates.get("qwen")
assert(t, "qwen template required")

local tests = {}

-- 1. Backend init + model load + context (no decode)
function tests.init_and_load()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local ctx = model:context({ n_ctx = 256, n_batch = 128 })
	assert(ctx.ctx ~= nil and model.model ~= nil)
end

-- 2. Tokenize only (add_bos true and false) and token_to_piece
function tests.tokenize_only()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local tokens_bos = model:tokenize("Hi", true)
	local tokens_no_bos = model:tokenize("Hi", false)
	assert(#tokens_bos >= 1 and #tokens_no_bos >= 1)
	local piece = model:token_to_piece(tokens_bos[1])
	assert(type(piece) == "string", "model:token_to_piece returns string")
end

-- 3. Single-token decode (like test_bindings)
function tests.decode_one_token()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local ctx = model:context({ n_ctx = 256, n_batch = 128 })
	local tokens = model:tokenize("Hello", true)
	assert(#tokens >= 1)
	local err = ctx:decode_tokens({ tokens[1] })
	assert(err == 0, "decode_tokens one token failed: " .. tostring(err))
	local logits = ctx:logits()
	assert(logits ~= nil, "logits nil")
end

-- 4. Decode full short prompt
function tests.decode_full_prompt()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local ctx = model:context({ n_ctx = 256, n_batch = 256 })
	local prompt = "Hello world"
	local tokens = model:tokenize(prompt, false)
	assert(#tokens >= 1)
	local err = ctx:decode_tokens(tokens)
	assert(err == 0, "decode_tokens full prompt failed: " .. tostring(err))
	local logits = ctx:logits()
	assert(logits ~= nil)
end

-- 5. Prefill + decode_one loop (no sampler) for a few steps
function tests.decode_one_loop()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local ctx = model:context({ n_ctx = 512, n_batch = 256 })
	local tokens = model:tokenize("Say hi", false)
	assert(#tokens >= 1)
	local err = ctx:decode_tokens(tokens)
	assert(err == 0)
	local n_past = #tokens
	-- Decode 3 dummy tokens (we don't sample; use a fixed token like first of vocab)
	local vocab = model:vocab()
	local dummy = 0
	for _ = 1, 3 do
		err = ctx:decode_one(dummy, n_past)
		assert(err == 0, "decode_one failed at n_past=" .. tostring(n_past))
		n_past = n_past + 1
	end
end

-- 6. Full sampler path: prefill, accept, sample, decode_one, repeat a few times
function tests.sampler_loop()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local ctx = model:context({ n_ctx = 2048, n_batch = 512 })
	local vocab = model:vocab()
	local eos = llama.llama_vocab_eos(vocab)

	-- Match chat.lua first turn exactly (system + user) to avoid sampler path differences
	local prompt = lluama.chat_templates.format_conversation(
		{ { role = "user", content = "Hi" } },
		"You are a helpful assistant. Answer concisely.",
		t
	)
	local tokens = model:tokenize(prompt, false)
	assert(#tokens >= 1)
	-- Set sampler before decode (same as chat.lua) so backend associates logits with seq 0
	local sampler = lluama.Sampler({ temp = 0.7, seed = 12345 })
	ctx:set_sampler(sampler)
	local err = ctx:decode_tokens(tokens)
	assert(err == 0)
	for i = 1, #tokens do
		sampler:accept(tokens[i])
	end

	local n_past = #tokens
	local logits_idx = #tokens - 1
	local max_steps = 20
	local reply = ""
	for step = 1, max_steps do
		local next_token = sampler:sample(ctx.ctx, logits_idx)
		if next_token == eos then break end
		reply = reply .. model:token_to_piece(next_token)
		sampler:accept(next_token)
		err = ctx:decode_one(next_token, n_past)
		assert(err == 0, "decode_one failed at step " .. tostring(step) .. " n_past=" .. tostring(n_past))
		n_past = n_past + 1
		logits_idx = 0
	end
	assert(type(reply) == "string", "token_to_piece coverage")
	assert(not reply:match("%<|%s*$"), "reply should not end with <|")
end

-- 7. Two turns on same context (two prefills + short sample each)
function tests.two_turns()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local ctx = model:context({ n_ctx = 1024, n_batch = 512 })
	local vocab = model:vocab()
	local eos = llama.llama_vocab_eos(vocab)
	local sampler = lluama.Sampler({ temp = 0.5, seed = 42 })
	ctx:set_sampler(sampler)

	local n_past = 0
	for turn = 1, 2 do
		local prompt = lluama.chat_templates.format_conversation(
			{ { role = "user", content = "Say number " .. tostring(turn) } },
			nil,
			t
		)
		local tokens = model:tokenize(prompt, false)
		assert(#tokens >= 1)
		local err = ctx:decode_tokens(tokens, n_past)
		assert(err == 0, "turn " .. turn .. " decode_tokens failed")
		for i = 1, #tokens do sampler:accept(tokens[i]) end
		n_past = n_past + #tokens
		local logits_idx = #tokens - 1
		local turn_reply = ""
		for _ = 1, 5 do
			local next_token = sampler:sample(ctx.ctx, logits_idx)
			if next_token == eos then break end
			turn_reply = turn_reply .. model:token_to_piece(next_token)
			sampler:accept(next_token)
			err = ctx:decode_one(next_token, n_past)
			assert(err == 0, "turn " .. turn .. " decode_one failed")
			n_past = n_past + 1
			logits_idx = 0
		end
		assert(type(turn_reply) == "string", "token_to_piece in two_turns")
	end
end

-- 8. Chat-style multi-turn: first turn system+user, second turn user message only (like chat.lua)
-- Catches missing n_past on second decode_tokens (would get Decode error: -1).
function tests.chat_style_turns()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local ctx = model:context({ n_ctx = 2048, n_batch = 512 })
	local vocab = model:vocab()
	local eos = llama.llama_vocab_eos(vocab)
	local sampler = lluama.Sampler({ temp = 0.5, seed = 123 })
	ctx:set_sampler(sampler)

	-- Turn 1: like first user message in chat (system + user "Hello?")
	local prompt1 = lluama.chat_templates.format_conversation(
		{ { role = "user", content = "Hello?" } },
		"You are a helpful assistant. Answer concisely.",
		t
	)
	local tokens1 = model:tokenize(prompt1, false)
	assert(#tokens1 >= 1)
	local err = ctx:decode_tokens(tokens1, 0)
	assert(err == 0, "turn 1 decode_tokens failed")
	for i = 1, #tokens1 do sampler:accept(tokens1[i]) end
	local n_past = #tokens1
	local logits_idx = #tokens1 - 1
	local reply1 = ""
	for _ = 1, 10 do
		local next_token = sampler:sample(ctx.ctx, logits_idx)
		if next_token == eos then break end
		reply1 = reply1 .. model:token_to_piece(next_token)
		sampler:accept(next_token)
		err = ctx:decode_one(next_token, n_past)
		assert(err == 0, "turn 1 decode_one failed")
		n_past = n_past + 1
		logits_idx = 0
	end

	-- Turn 2: like second user message in chat (user_start .. line .. user_end .. assistant_start)
	local prompt2 = t.user_start .. "Write a python script to plot a line" .. t.user_end .. t.assistant_start
	local tokens2 = model:tokenize(prompt2, false)
	assert(#tokens2 >= 1)
	err = ctx:decode_tokens(tokens2, n_past)
	assert(err == 0, "turn 2 decode_tokens failed (forgot n_past?)")
	for i = 1, #tokens2 do sampler:accept(tokens2[i]) end
	n_past = n_past + #tokens2
	logits_idx = #tokens2 - 1
	local reply2 = ""
	for _ = 1, 5 do
		local next_token = sampler:sample(ctx.ctx, logits_idx)
		if next_token == eos then break end
		reply2 = reply2 .. model:token_to_piece(next_token)
		sampler:accept(next_token)
		err = ctx:decode_one(next_token, n_past)
		assert(err == 0, "turn 2 decode_one failed")
		n_past = n_past + 1
		logits_idx = 0
	end
	assert(not reply2:match("%<|%s*$"), "chat_style turn 2 reply should not end with <|")
end

-- 9. ChatSession: one prompt + generate (same behavior as chat.lua one turn)
function tests.chat_session_one_turn()
	local backend = lluama.Backend()
	backend:init()
	local session = lluama.ChatSession(backend, model_path, {
		template = "qwen",
		system_prompt = "You are a helpful assistant. Answer concisely.",
		temp = 0.5,
		seed = 99,
	})
	local err = session:prompt("Say hello in one word.")
	assert(err == 0, "ChatSession:prompt failed")
	local reply = session:generate(20, nil)  -- no stream, return full reply
	assert(type(reply) == "string")
	assert(not reply:match("%<|%s*$"), "reply should not end with <|")
end

-- 10. ChatSession: two turns (prompt+generate, prompt+generate) for full session coverage
function tests.chat_session_two_turns()
	local backend = lluama.Backend()
	backend:init()
	local session = lluama.ChatSession(backend, model_path, {
		template = "qwen",
		system_prompt = "You are helpful.",
		temp = 0.5,
		seed = 42,
	})
	local err = session:prompt("What is 1+1? Reply with one number.")
	assert(err == 0)
	local reply1 = session:generate(15, nil)
	assert(type(reply1) == "string")
	err = session:prompt("Now say that number in words.")
	assert(err == 0)
	local reply2 = session:generate(15, nil)
	assert(type(reply2) == "string")
	assert(not reply2:match("%<|%s*$"), "second reply should not end with <|")
end

-- 11. Model accessors
function tests.model_accessors()
	local backend = lluama.Backend()
	backend:init()
	local model = lluama.Model(backend, model_path)
	local v = model:vocab()
	assert(v ~= nil)
	local n = model:n_params()
	assert(n ~= nil and (tonumber(n) or n) > 0)
	local d = model:desc()
	assert(d ~= nil and type(tostring(d)) == "string")
end

local test_order = {
	"init_and_load",
	"tokenize_only",
	"decode_one_token",
	"decode_full_prompt",
	"decode_one_loop",
	"sampler_loop",
	"two_turns",
	"chat_style_turns",
	"chat_session_one_turn",
	"chat_session_two_turns",
	"model_accessors",
}

local total_iters = 0
local total_tests = 0
local total_fails = 0
local failed_detail = {}

for iter = 1, loop_count do
	total_iters = iter
	for _, name in ipairs(test_order) do
		if only_name and only_name ~= name then goto continue end
		local fn = tests[name]
		if not fn then goto continue end
		total_tests = total_tests + 1
		local ok, err = pcall(fn)
		if not ok then
			total_fails = total_fails + 1
			local msg = string.format("iter %d test %s: %s", iter, name, tostring(err))
			table.insert(failed_detail, msg)
			io.stderr:write("[FAIL] " .. msg .. "\n")
			io.stderr:flush()
		end
		::continue::
	end
	if loop_count > 1 then
		if iter % 5 == 0 or iter == 1 then
			io.write(string.format("  iter %d\n", iter))
			io.flush()
		end
	end
end

print("")
print(string.format("Iterations: %d  Tests run: %d  Failed: %d", total_iters, total_tests, total_fails))
if total_fails > 0 then
	for _, msg in ipairs(failed_detail) do
		io.stderr:write(msg .. "\n")
	end
	os.exit(1)
end
print("All integration tests passed.")
