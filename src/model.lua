-- Model: holds backend (keeps it alive), .model (cdata). __gc frees model.
-- Methods error with "lluama: model already freed" if used after __gc.

local ffi = require("ffi")

local DEFAULT_TOKENIZE_CAP = 4096
local PIECE_BUF_SIZE = 256
local DESC_BUF_SIZE = 256

return function(llama, lluama)
	local function assert_model(self)
		if self.model == nil then
			error("lluama: model already freed")
		end
	end

	local Model_mt = {
		__gc = function(self)
			if self.model ~= nil then
				llama.llama_free_model(self.model)
				self.model = nil
				self._vocab = nil
				self._tokenize_buf = nil
				self._tokenize_buf_cap = nil
				self._piece_buf = nil
				self._desc_buf = nil
				self._detokenize_buf = nil
				self._detokenize_buf_cap = nil
				self._detokenize_token_buf = nil
				self._detokenize_token_cap = nil
				self._meta_key_buf = nil
				self._meta_val_buf = nil
			end
		end,
		__index = {},
	}
	Model_mt.__index = Model_mt

	-- opts: n_ctx, n_batch, n_threads, n_threads_batch
	function Model_mt.context(self, opts)
		assert_model(self)
		opts = opts or {}
		local cparams = llama.llama_context_default_params()
		cparams.n_ctx = opts.n_ctx or 512
		cparams.n_batch = opts.n_batch or 256
		if opts.n_threads ~= nil then cparams.n_threads = opts.n_threads end
		if opts.n_threads_batch ~= nil then cparams.n_threads_batch = opts.n_threads_batch end
		local ctx = llama.llama_new_context_with_model(self.model, cparams)
		if ctx == nil then
			error("lluama: failed to create context")
		end
		return lluama.Context(self, ctx)
	end

	-- text (string), add_bos (bool, default true), parse_special (bool, default false).
	-- Grows buffer and retries when token count fills buffer so long text never truncates.
	function Model_mt.tokenize(self, text, add_bos, parse_special)
		assert_model(self)
		if add_bos == nil then add_bos = true end
		if parse_special == nil then parse_special = false end
		local vocab = self:vocab()
		local cap = self._tokenize_buf_cap or DEFAULT_TOKENIZE_CAP
		if not self._tokenize_buf or self._tokenize_buf_cap < cap then
			self._tokenize_buf_cap = cap
			self._tokenize_buf = ffi.new("int32_t[?]", cap)
		end
		local n
		repeat
			local buf = self._tokenize_buf
			n = llama.llama_tokenize(vocab, text, #text, buf, cap, add_bos, parse_special)
			if n < 0 then
				error("lluama: tokenize failed")
			end
			if n >= cap then
				cap = cap * 2
				self._tokenize_buf_cap = cap
				self._tokenize_buf = ffi.new("int32_t[?]", cap)
			end
		until n < cap
		local out = {}
		for i = 0, n - 1 do
			out[i + 1] = self._tokenize_buf[i]
		end
		return out
	end

	-- Decode a single token id to its string representation (piece).
	-- Reuses a single buffer. special (bool, default false) = include special token text.
	function Model_mt.token_to_piece(self, token, special)
		assert_model(self)
		if special == nil then special = false end
		local vocab = self:vocab()
		if not self._piece_buf then
			self._piece_buf = ffi.new("char[?]", PIECE_BUF_SIZE)
		end
		local buf = self._piece_buf
		local n = llama.llama_token_to_piece(vocab, token, buf, PIECE_BUF_SIZE, 0, special)
		if n <= 0 then return "" end
		if n >= PIECE_BUF_SIZE then
			local big = ffi.new("char[?]", n + 1)
			n = llama.llama_token_to_piece(vocab, token, big, n + 1, 0, special)
			if n <= 0 then return "" end
			return ffi.string(big, n)
		end
		return ffi.string(buf, n)
	end

	-- Vocab (cached).
	function Model_mt.vocab(self)
		assert_model(self)
		if self._vocab == nil then
			self._vocab = llama.llama_model_get_vocab(self.model)
		end
		return self._vocab
	end

	function Model_mt.n_params(self)
		assert_model(self)
		return llama.llama_model_n_params(self.model)
	end

	function Model_mt.size(self)
		assert_model(self)
		return llama.llama_model_size(self.model)
	end

	function Model_mt.save_to_file(self, path)
		assert_model(self)
		llama.llama_model_save_to_file(self.model, path)
	end

	-- Architecture / query
	function Model_mt.n_ctx_train(self)
		assert_model(self)
		return llama.llama_model_n_ctx_train(self.model)
	end
	function Model_mt.n_embd(self)
		assert_model(self)
		return llama.llama_model_n_embd(self.model)
	end
	function Model_mt.n_embd_inp(self)
		assert_model(self)
		return llama.llama_model_n_embd_inp(self.model)
	end
	function Model_mt.n_embd_out(self)
		assert_model(self)
		return llama.llama_model_n_embd_out(self.model)
	end
	function Model_mt.n_layer(self)
		assert_model(self)
		return llama.llama_model_n_layer(self.model)
	end
	function Model_mt.n_head(self)
		assert_model(self)
		return llama.llama_model_n_head(self.model)
	end
	function Model_mt.n_head_kv(self)
		assert_model(self)
		return llama.llama_model_n_head_kv(self.model)
	end
	function Model_mt.n_swa(self)
		assert_model(self)
		return llama.llama_model_n_swa(self.model)
	end
	function Model_mt.rope_type(self)
		assert_model(self)
		return llama.llama_model_rope_type(self.model)
	end
	function Model_mt.rope_freq_scale_train(self)
		assert_model(self)
		return llama.llama_model_rope_freq_scale_train(self.model)
	end
	function Model_mt.has_encoder(self)
		assert_model(self)
		return llama.llama_model_has_encoder(self.model)
	end
	function Model_mt.has_decoder(self)
		assert_model(self)
		return llama.llama_model_has_decoder(self.model)
	end
	function Model_mt.decoder_start_token(self)
		assert_model(self)
		return llama.llama_model_decoder_start_token(self.model)
	end
	function Model_mt.is_recurrent(self)
		assert_model(self)
		return llama.llama_model_is_recurrent(self.model)
	end
	function Model_mt.is_hybrid(self)
		assert_model(self)
		return llama.llama_model_is_hybrid(self.model)
	end
	function Model_mt.is_diffusion(self)
		assert_model(self)
		return llama.llama_model_is_diffusion(self.model)
	end

	-- Classification (encoder models)
	function Model_mt.n_cls_out(self)
		assert_model(self)
		return llama.llama_model_n_cls_out(self.model)
	end
	function Model_mt.cls_label(self, i)
		assert_model(self)
		local p = llama.llama_model_cls_label(self.model, i)
		return p ~= nil and ffi.string(p) or nil
	end

	-- Model meta (KV from file)
	function Model_mt.meta_count(self)
		assert_model(self)
		return llama.llama_model_meta_count(self.model)
	end
	function Model_mt.meta_key_by_index(self, i)
		assert_model(self)
		if not self._meta_key_buf then
			self._meta_key_buf = ffi.new("char[?]", 128)
		end
		local n = llama.llama_model_meta_key_by_index(self.model, i, self._meta_key_buf, 128)
		return n > 0 and ffi.string(self._meta_key_buf, n) or nil
	end
	function Model_mt.meta_val_str_by_index(self, i)
		assert_model(self)
		if not self._meta_val_buf then
			self._meta_val_buf = ffi.new("char[?]", 256)
		end
		local n = llama.llama_model_meta_val_str_by_index(self.model, i, self._meta_val_buf, 256)
		return n > 0 and ffi.string(self._meta_val_buf, n) or nil
	end

	-- Built-in chat template from model file (name e.g. "chatml")
	function Model_mt.chat_template(self, name)
		assert_model(self)
		local p = name and llama.llama_model_chat_template(self.model, name) or nil
		return p ~= nil and ffi.string(p) or nil
	end

	-- Vocab special token ids (convenience)
	function Model_mt.bos(self) return llama.llama_vocab_bos(self:vocab()) end
	function Model_mt.eos(self) return llama.llama_vocab_eos(self:vocab()) end
	function Model_mt.eot(self) return llama.llama_vocab_eot(self:vocab()) end
	function Model_mt.sep(self) return llama.llama_vocab_sep(self:vocab()) end
	function Model_mt.nl(self) return llama.llama_vocab_nl(self:vocab()) end
	function Model_mt.pad(self) return llama.llama_vocab_pad(self:vocab()) end
	function Model_mt.mask(self) return llama.llama_vocab_mask(self:vocab()) end
	function Model_mt.vocab_n_tokens(self) return llama.llama_vocab_n_tokens(self:vocab()) end
	function Model_mt.vocab_type(self) return llama.llama_vocab_type(self:vocab()) end
	function Model_mt.vocab_get_text(self, token)
		local p = llama.llama_vocab_get_text(self:vocab(), token)
		return p ~= nil and ffi.string(p) or nil
	end
	function Model_mt.vocab_get_add_bos(self) return llama.llama_vocab_get_add_bos(self:vocab()) end
	function Model_mt.vocab_get_add_eos(self) return llama.llama_vocab_get_add_eos(self:vocab()) end

	-- Detokenize: token ids -> string. remove_special, unparse_special (default false).
	function Model_mt.detokenize(self, token_ids, remove_special, unparse_special)
		assert_model(self)
		if remove_special == nil then remove_special = false end
		if unparse_special == nil then unparse_special = false end
		local n = #token_ids
		if n == 0 then return "" end
		local vocab = self:vocab()
		local cap = 4096
		if not self._detokenize_buf or self._detokenize_buf_cap < cap then
			self._detokenize_buf_cap = cap
			self._detokenize_buf = ffi.new("char[?]", cap)
		end
		local buf = self._detokenize_buf
		if not self._detokenize_token_buf or self._detokenize_token_cap < n then
			self._detokenize_token_cap = math.max(n, 256)
			self._detokenize_token_buf = ffi.new("int32_t[?]", self._detokenize_token_cap)
		end
		local token_buf = self._detokenize_token_buf
		for i = 0, n - 1 do
			token_buf[i] = token_ids[i + 1]
		end
		local len = llama.llama_detokenize(vocab, token_buf, n, buf, cap, remove_special, unparse_special)
		if len <= 0 then return "" end
		if len >= cap then
			self._detokenize_buf_cap = len + 1
			self._detokenize_buf = ffi.new("char[?]", self._detokenize_buf_cap)
			buf = self._detokenize_buf
			len = llama.llama_detokenize(vocab, token_buf, n, buf, len + 1, remove_special, unparse_special)
			if len <= 0 then return "" end
		end
		return ffi.string(buf, len)
	end

	function Model_mt.desc(self)
		assert_model(self)
		if not self._desc_buf then
			self._desc_buf = ffi.new("char[?]", DESC_BUF_SIZE)
		end
		local buf = self._desc_buf
		local n = llama.llama_model_desc(self.model, buf, DESC_BUF_SIZE)
		if n <= 0 then return "" end
		return ffi.string(buf, n > DESC_BUF_SIZE and DESC_BUF_SIZE or n)
	end

	return function(backend, path, opts)
		opts = opts or {}
		local mparams = llama.llama_model_default_params()
		if opts.n_gpu_layers ~= nil then mparams.n_gpu_layers = opts.n_gpu_layers end
		if opts.use_mmap ~= nil then mparams.use_mmap = opts.use_mmap end
		if opts.use_mlock ~= nil then mparams.use_mlock = opts.use_mlock end
		local model = llama.llama_load_model_from_file(path, mparams)
		if model == nil then
			error("lluama: failed to load model: " .. tostring(path))
		end
		return setmetatable({
			model = model,
			backend = backend,
		}, Model_mt)
	end
end
