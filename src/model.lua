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
