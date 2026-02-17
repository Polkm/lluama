-- ChatSession: high-level chat loop. Uses native llama_chat_apply_template only.
-- See: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
-- Use :prompt(user_message) then :generate(max_tokens, stream_cb) for each turn.

return function(lluama)
	local llama = lluama.llama
	local chat_native = lluama.chat_native

	-- Stop-string helpers (vocab-derived stop_sequences). Longest-first order for clean_response.
	local function clean_response(text, stop_sequences)
		for _, seq in ipairs(stop_sequences) do
			local pos = text:find(seq, 1, true)
			if pos then
				text = text:sub(1, pos - 1)
			end
		end
		return text:match("^%s*(.-)%s*$") or text
	end

	local function check_stop(text, stop_sequences)
		for _, seq in ipairs(stop_sequences) do
			local pos = text:find(seq, 1, true)
			if pos then
				return true, text:sub(1, pos - 1)
			end
		end
		return false, text
	end

	local function trim_trailing_stop_prefix(text, stop_sequences)
		local out = text
		local changed = true
		while changed do
			changed = false
			for _, seq in ipairs(stop_sequences) do
				for k = #seq - 1, 1, -1 do
					if #out >= k and out:sub(-k) == seq:sub(1, k) then
						out = out:sub(1, #out - k)
						changed = true
						break
					end
				end
				if changed then break end
			end
		end
		return out
	end

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
		local messages
		if self._first_turn then
			self._first_turn = false
			messages = {}
			if self._system_prompt and #self._system_prompt > 0 then
				messages[1] = { role = "system", content = self._system_prompt }
			end
			messages[#messages + 1] = { role = "user", content = user_message }
		else
			messages = { { role = "user", content = user_message } }
		end
		local prompt = chat_native.chat_apply_template(self._native_template_str, messages, true)
		if not prompt then
			error("ChatSession: native chat_apply_template failed")
		end
		-- When using a grammar, ensure prompt ends with newline so grammars that use
		-- prefix ::= .* "\n" see a complete prefix before the constrained part.
		if self._grammar and prompt:sub(-1) ~= "\n" then
			prompt = prompt .. "\n"
		end
		local tokens = self.model:tokenize(prompt, false)
		if #tokens == 0 then return 0 end
		local err = self.ctx:decode_tokens(tokens, self._n_past)
		if err ~= 0 then return err end
		-- When using a grammar (e.g. JSON with prefix), reset then accept the prompt so the grammar
		-- sees the prefix; generate() will then only allow the constrained part (e.g. JSON object).
		if self._grammar then
			self.sampler:reset()
		end
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
		local stop_sequences = self._stop_sequences
		local n_tokens = self._last_n_tokens
		local logits_idx = n_tokens - 1
		local reply = ""
		local printed_len = 0
		-- Grammar state was set in prompt() (we accepted the prefix); do not reset here.

		for _ = 1, max_tokens do
			if self._on_before_sample then
				self._on_before_sample(self.ctx, logits_idx)
			end
			local next_token = self.sampler:sample(ctx, logits_idx)
			if next_token == eos_id then break end
			if self._stop_token_ids[next_token] then
				local cleaned = clean_response(reply, stop_sequences)
				cleaned = trim_trailing_stop_prefix(cleaned, stop_sequences)
				if stream_cb and #cleaned > printed_len then
					stream_cb(cleaned:sub(printed_len + 1))
				end
				reply = cleaned
				break
			end

			local piece = self.model:token_to_piece(next_token)
			reply = reply .. piece
			local should_stop, cleaned = check_stop(reply, stop_sequences)
			if should_stop then
				cleaned = clean_response(cleaned, stop_sequences)
				if stream_cb and #cleaned > printed_len then
					stream_cb(cleaned:sub(printed_len + 1))
				end
				reply = cleaned
				break
			end

			if stream_cb and not should_skip_print(piece, reply, stop_sequences) then
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
		local stop_sequences = model:stop_strings_from_vocab()
		local ctx = model:context({
			n_ctx = opts.n_ctx or 2048,
			n_batch = opts.n_batch or 512,
		})
		local sampler_opts = {
			temp = opts.temp or 0.7,
			seed = opts.seed or 12345,
		}
		if opts.grammar then
			sampler_opts.grammar = opts.grammar
			sampler_opts.grammar_root = opts.grammar_root
		end
		local sampler = lluama.Sampler(sampler_opts, opts.grammar and model or nil)
		ctx:set_sampler(sampler)

		local stop_token_ids = {}
		for _, seq in ipairs(stop_sequences) do
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
			_stop_sequences = stop_sequences,
			_system_prompt = opts.system_prompt or "",
			_n_past = 0,
			_first_turn = true,
			_last_n_tokens = 0,
			_stop_token_ids = stop_token_ids,
			_eos_id = eos_id,
			_grammar = opts.grammar and true or nil,
			_native_template_str = native_tmpl,
			_on_before_sample = opts.on_before_sample,
		}, ChatSession_mt)
	end
end
