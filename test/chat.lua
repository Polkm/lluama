-- CLI chat: back-and-forth with the model using a chat template. Run from repo root.
-- Usage: luajit test/chat.lua [model_path]
-- Default model: models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf (Qwen/ChatML format)

package.path = "./?.lua;" .. package.path
local ffi = require("ffi")
local lluama = require("src.lluama")

local MODEL_PATH = arg[1] or "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"
local MAX_NEW_TOKENS = 1024
local TEMP = 0.7
local SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

-- Chat template: Qwen and similar use ChatML. Use qwen or chatml for <|im_start|>user/assistant format.
local t = lluama.chat_templates.qwen or lluama.chat_templates.chatml
if not t then
	error("chat.lua requires chat_templates.qwen or chat_templates.chatml")
end

local function token_to_piece(llama, vocab, token)
	local buf = ffi.new("char[64]")
	local n = llama.llama_token_to_piece(vocab, token, buf, 64, 0, false)
	if n <= 0 then return "" end
	return ffi.string(buf, n)
end

local function main()
	local backend = lluama.Backend()
	backend:init()

	local model = lluama.Model(backend, MODEL_PATH)
	local ctx = model:context({ n_ctx = 2048, n_batch = 512 })

	local llama = lluama.llama
	local vocab = llama.llama_model_get_vocab(model.model)

	-- Sampler and context APIs (decode_one uses internal batch; Sampler __gc frees chain)
	local sampler = lluama.Sampler({ temp = TEMP, seed = 12345 })
	ctx:set_sampler(sampler)

	local eos_id = llama.llama_vocab_eos(vocab)
	-- Stop on token id as well: special tokens (e.g. <|im_end|>) may decode to "" so we'd never see them in reply.
	local stop_token_ids = {}
	for _, seq in ipairs(t.stop_sequences) do
		for _, id in ipairs(model:tokenize(seq, false)) do
			stop_token_ids[id] = true
		end
	end
	local n_past = 0
	local first_turn = true

	print("Ready. Type a message and press Enter (empty to quit).")
	while true do
		io.write("You: ")
		io.flush()
		local line = io.read("*l")
		if not line then break end
		line = line:match("^%s*(.-)%s*$") or line  -- trim
		if line == "" then break end

		-- Build prompt with chat template so the model sees user/assistant turns
		local prompt
		if first_turn then
			prompt = lluama.chat_templates.format_conversation(
				{ { role = "user", content = line } },
				SYSTEM_PROMPT,
				t
			)
			first_turn = false
		else
			prompt = t.user_start .. line .. t.user_end .. t.assistant_start
		end
		local tokens = model:tokenize(prompt, false)  -- template has its own special tokens
		if #tokens == 0 then goto continue end

		-- Prefill at continuing positions (second+ turn must use n_past)
		local err = ctx:decode_tokens(tokens, n_past)
		if err ~= 0 then
			io.stderr:write("Decode error: " .. tostring(err) .. "\n")
			goto continue
		end
		n_past = n_past + #tokens

		-- Accept prompt tokens into sampler (e.g. for repeat penalty)
		for i = 1, #tokens do
			sampler:accept(tokens[i])
		end

		-- Sample loop: accumulate reply and stop when we hit a template stop (e.g. <|im_end|>)
		io.write("Model: ")
		io.flush()
		local n = 0
		local logits_idx = #tokens - 1
		local reply = ""
		local printed_len = 0
		while n < MAX_NEW_TOKENS do
			local next_token = sampler:sample(ctx.ctx, logits_idx)
			if next_token == eos_id then break end
			if stop_token_ids[next_token] then
				-- Stop token (e.g. <|im_end|>) may decode to ""; stop on token id so we don't miss it
				local cleaned = lluama.chat_templates.clean_response(reply, t)
				if #cleaned > printed_len then
					io.write(cleaned:sub(printed_len + 1))
					io.flush()
				end
				break
			end

			local piece = token_to_piece(llama, vocab, next_token)
			reply = reply .. piece
			-- Check full reply for stop sequence (can span multiple tokens)
			local should_stop, cleaned = lluama.chat_templates.check_stop(reply, t)
			if should_stop then
				cleaned = lluama.chat_templates.clean_response(cleaned, t)
				if #cleaned > printed_len then
					io.write(cleaned:sub(printed_len + 1))
					io.flush()
				end
				break
			end

			-- Don't print if piece or end of reply is a prefix of any stop sequence (avoids showing "<|im_end" or "<|im_end|")
			local skip_print = false
			for _, seq in ipairs(t.stop_sequences) do
				if #piece <= #seq and seq:sub(1, #piece) == piece then
					skip_print = true
					break
				end
				for k = 1, #seq - 1 do
					if #reply >= k and reply:sub(-k) == seq:sub(1, k) then
						skip_print = true
						break
					end
				end
			end
			if not skip_print then
				io.write(piece)
				io.flush()
				printed_len = #reply
			end
			n = n + 1

			sampler:accept(next_token)
			err = ctx:decode_one(next_token, n_past)
			if err ~= 0 then break end
			n_past = n_past + 1
			logits_idx = 0  -- next decode has batch size 1
		end
		print()
		::continue::
	end
end

local ok, err = pcall(main)
if not ok then
	io.stderr:write(tostring(err) .. "\n")
	os.exit(1)
end
