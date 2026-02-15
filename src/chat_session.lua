-- ChatSession: high-level chat loop. Handles template, decode positions, sampler, and stop logic.
-- Use :prompt(user_message) then :generate(max_tokens, stream_cb) for each turn.

return function(lluama)
	local llama = lluama.llama
	local templates = lluama.chat_templates

	local function should_skip_print(piece, reply, stop_sequences)
		for _, seq in ipairs(stop_sequences) do
			if #piece <= #seq and seq:sub(1, #piece) == piece then return true end
			for k = 1, #seq - 1 do
				if #reply >= k and reply:sub(-k) == seq:sub(1, k) then return true end
			end
		end
		return false
	end

	local ChatSession_mt = {}
	ChatSession_mt.__index = ChatSession_mt

	function ChatSession_mt:prompt(user_message)
		local prompt
		if self._first_turn then
			prompt = templates.format_conversation(
				{ { role = "user", content = user_message } },
				self._system_prompt,
				self._template
			)
			self._first_turn = false
		else
			local t = self._template
			prompt = t.user_start .. user_message .. t.user_end .. t.assistant_start
		end
		local tokens = self.model:tokenize(prompt, false)
		if #tokens == 0 then return 0 end
		local err = self.ctx:decode_tokens(tokens, self._n_past)
		if err ~= 0 then return err end
		for i = 1, #tokens do
			self.sampler:accept(tokens[i])
		end
		self._n_past = self._n_past + #tokens
		self._last_n_tokens = #tokens
		return 0
	end

	function ChatSession_mt:generate(max_tokens, stream_cb)
		max_tokens = max_tokens or 1024
		local ctx = self.ctx.ctx
		local eos_id = self._eos_id
		local t = self._template
		local n_tokens = self._last_n_tokens
		local logits_idx = n_tokens - 1
		local reply = ""
		local printed_len = 0

		for _ = 1, max_tokens do
			local next_token = self.sampler:sample(ctx, logits_idx)
			if next_token == eos_id then break end
			if self._stop_token_ids[next_token] then
				local cleaned = templates.clean_response(reply, t)
				cleaned = templates.trim_trailing_stop_prefix(cleaned, t)
				if stream_cb and #cleaned > printed_len then
					stream_cb(cleaned:sub(printed_len + 1))
				end
				reply = cleaned
				break
			end

			local piece = self.model:token_to_piece(next_token)
			reply = reply .. piece
			local should_stop, cleaned = templates.check_stop(reply, t)
			if should_stop then
				cleaned = templates.clean_response(cleaned, t)
				if stream_cb and #cleaned > printed_len then
					stream_cb(cleaned:sub(printed_len + 1))
				end
				reply = cleaned
				break
			end

			if stream_cb and not should_skip_print(piece, reply, t.stop_sequences) then
				stream_cb(piece)
				printed_len = #reply
			end

			self.sampler:accept(next_token)
			local err = self.ctx:decode_one(next_token, self._n_past)
			if err ~= 0 then break end
			self._n_past = self._n_past + 1
			logits_idx = 0
		end
		return reply
	end

	return function(backend, model_path, opts)
		opts = opts or {}
		local template_name = opts.template or "qwen"
		local t = templates.get(template_name)
		if not t then
			error("ChatSession: template '" .. template_name .. "' not found")
		end
		local model = lluama.Model(backend, model_path)
		local ctx = model:context({
			n_ctx = opts.n_ctx or 2048,
			n_batch = opts.n_batch or 512,
		})
		local sampler = lluama.Sampler({
			temp = opts.temp or 0.7,
			seed = opts.seed or 12345,
		})
		ctx:set_sampler(sampler)

		local stop_token_ids = {}
		for _, seq in ipairs(t.stop_sequences) do
			local ids = model:tokenize(seq, false)
			if #ids > 0 then stop_token_ids[ids[#ids]] = true end
		end

		local vocab = model:vocab()
		local eos_id = llama.llama_vocab_eos(vocab)

		return setmetatable({
			backend = backend,
			model = model,
			ctx = ctx,
			sampler = sampler,
			_template = t,
			_system_prompt = opts.system_prompt or "",
			_n_past = 0,
			_first_turn = true,
			_last_n_tokens = 0,
			_stop_token_ids = stop_token_ids,
			_eos_id = eos_id,
		}, ChatSession_mt)
	end
end
