-- ChatSession: high-level chat loop mirroring llama.cpp examples/simple-chat/simple-chat.cpp.
-- Maintains a full message list, applies the chat template each turn, and only feeds the
-- new slice (formatted[prev_len..new_len]) to the model. Uses llama_vocab_is_eog for stop.
-- See: https://github.com/ggml-org/llama.cpp/blob/master/examples/simple-chat/simple-chat.cpp

return function(lluama)
	local llama = lluama.llama
	local chat_native = lluama.chat_native

	local ChatSession_mt = {}
	ChatSession_mt.__index = ChatSession_mt

	-- Add user message to the list and prepare the prompt slice for this turn.
	-- Mirrors simple-chat: messages.push_back(user), apply_template(..., true), prompt = formatted[prev_len..new_len].
	function ChatSession_mt:prompt(user_message)
		self._messages[#self._messages + 1] = { role = "user", content = user_message }
		local formatted = chat_native.chat_apply_template(self._native_template_str, self._messages, true)
		if not formatted then
			error("ChatSession: chat_apply_template failed")
		end
		-- Prompt to generate = only the new part (user turn + assistant prefix).
		local prompt_text = formatted:sub(self._prev_len + 1)
		if #prompt_text == 0 then
			return 0
		end
		-- is_first: true when context is empty (mirrors simple-chat llama_memory_seq_pos_max == -1).
		local n_ctx = self.ctx:n_ctx()
		local mem = self.ctx:get_memory()
		local n_ctx_used = (mem and llama.llama_memory_seq_pos_max(mem, 0) or -1) + 1
		local is_first = (n_ctx_used == 0)
		local tokens = self.model:tokenize(prompt_text, is_first, true)
		if #tokens == 0 then
			return 0
		end
		if n_ctx_used + #tokens > n_ctx then
			error("ChatSession: context size exceeded")
		end
		-- Decode prompt in chunks of at most n_batch for better perf and to respect context batch limit.
		local n_batch = self.ctx:n_batch()
		if self._grammar then
			self.sampler:reset()
		end
		local pos = 0
		while pos < #tokens do
			local chunk_end = math.min(pos + n_batch, #tokens)
			local chunk = {}
			for i = pos + 1, chunk_end do chunk[#chunk + 1] = tokens[i] end
			local err = self.ctx:decode_tokens(chunk, self._n_past)
			if err ~= 0 then
				return err
			end
			for i = 1, #chunk do
				self.sampler:accept(chunk[i])
			end
			self._n_past = self._n_past + #chunk
			pos = chunk_end
		end
		self._last_n_tokens = #tokens
		return 0
	end

	-- Generate loop: sample -> vocab_is_eog? -> token_to_piece/append -> accept + decode_one.
	-- Then add assistant response to messages and update prev_len (mirrors simple-chat).
	function ChatSession_mt:generate(max_tokens, stream_cb)
		max_tokens = max_tokens or 1024
		local ctx = self.ctx.ctx
		local model = self.model
		local n_tokens = self._last_n_tokens
		local logits_idx = n_tokens - 1
		local response = ""

		for _ = 1, max_tokens do
			if self._on_before_sample then
				self._on_before_sample(self.ctx, logits_idx)
			end
			local new_token_id = self.sampler:sample(ctx, logits_idx)

			if model:vocab_is_eog(new_token_id) then
				break
			end

			local piece = model:token_to_piece(new_token_id)
			if stream_cb then
				stream_cb(piece)
			end
			response = response .. piece

			self.sampler:accept(new_token_id)
			local err = self.ctx:decode_one(new_token_id, self._n_past)
			if err ~= 0 then
				break
			end
			self._n_past = self._n_past + 1
			logits_idx = 0
		end

		-- Add assistant response and update prev_len for next turn (simple-chat pattern).
		self._messages[#self._messages + 1] = { role = "assistant", content = response }
		local len = chat_native.chat_apply_template_length(self._native_template_str, self._messages, false)
		if len >= 0 then
			self._prev_len = len
		else
			local s = chat_native.chat_apply_template(self._native_template_str, self._messages, false)
			self._prev_len = s and #s or 0
		end

		return response
	end

	-- Resolve chat template: opts.template_string, or auto-detect from model.
	local function resolve_template(model, opts)
		if opts.template_string and #opts.template_string > 0 then
			return opts.template_string
		end
		local try_name = opts.template
		local is_auto = not try_name or try_name == "" or try_name == "auto"
		if not is_auto then
			local t = model:chat_template(try_name)
			if t and #t > 0 then return t end
		end
		for _, name in ipairs({ "default", "tokenizer", "tokenizer.chat_template" }) do
			local t = model:chat_template(name)
			if t and #t > 0 then return t end
		end
		local t = model:meta_val_str("tokenizer.chat_template")
		if t and #t > 0 then return t end
		for _, name in ipairs(lluama.chat_builtin_templates()) do
			if name and #name > 0 then
				local t = model:chat_template(name)
				if t and #t > 0 then return t end
			end
		end
		return nil
	end

	return function(backend, model_path, opts)
		opts = opts or {}
		local load_opts = {
			progress_callback = opts.progress_callback,
			kv_overrides = opts.kv_overrides,
			n_gpu_layers = opts.n_gpu_layers,
			use_mmap = opts.use_mmap,
			use_mlock = opts.use_mlock,
		}
		local model = lluama.Model(backend, model_path, load_opts)
		local native_tmpl = resolve_template(model, opts)
		if not native_tmpl or #native_tmpl == 0 then
			error("ChatSession: model has no chat template. Use a GGUF with tokenizer.chat_template, or pass opts.template_string.")
		end
		local n_ctx = opts.n_ctx or 2048
		local ctx = model:context({
			n_ctx = n_ctx,
			n_batch = opts.n_batch or n_ctx,
		})
		local sampler_opts = {
			temp = opts.temp or 0.8,
			seed = opts.seed or 12345,
			min_p = opts.min_p or 0.05,
		}
		if opts.grammar then
			sampler_opts.grammar = opts.grammar
			sampler_opts.grammar_root = opts.grammar_root
		end
		local sampler = lluama.Sampler(sampler_opts, opts.grammar and model or nil)
		ctx:set_sampler(sampler)

		-- Full message list (system optional, then user/assistant turns). prev_len = byte offset after last formatted convo.
		local messages = {}
		if opts.system_prompt and #opts.system_prompt > 0 then
			messages[1] = { role = "system", content = opts.system_prompt }
		end

		return setmetatable({
			backend = backend,
			model = model,
			ctx = ctx,
			sampler = sampler,
			_messages = messages,
			_prev_len = 0,
			_n_past = 0,
			_last_n_tokens = 0,
			_grammar = opts.grammar and true or nil,
			_native_template_str = native_tmpl,
			_on_before_sample = opts.on_before_sample,
		}, ChatSession_mt)
	end
end
