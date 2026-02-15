-- Model: holds backend (keeps it alive), .model (cdata). __gc frees model.

local ffi = require("ffi")

return function(llama, lluama)
	local Model_mt = {
		__gc = function(self)
			if self.model ~= nil then
				llama.llama_free_model(self.model)
				self.model = nil
			end
		end,
		__index = {},
	}
	Model_mt.__index = Model_mt

	function Model_mt.context(self, opts)
		opts = opts or {}
		local cparams = llama.llama_context_default_params()
		cparams.n_ctx = opts.n_ctx or 512
		cparams.n_batch = opts.n_batch or 256
		local ctx = llama.llama_new_context_with_model(self.model, cparams)
		if ctx == nil then
			error("lluama: failed to create context")
		end
		return lluama.Context(self, ctx)
	end

	function Model_mt.tokenize(self, text, add_bos)
		if add_bos == nil then add_bos = true end
		local vocab = llama.llama_model_get_vocab(self.model)
		local max_tokens = 4096
		local buf = ffi.new("int32_t[?]", max_tokens)
		local n = llama.llama_tokenize(vocab, text, #text, buf, max_tokens, add_bos, false)
		if n < 0 then
			error("lluama: tokenize failed")
		end
		local out = {}
		for i = 0, n - 1 do
			out[i + 1] = buf[i]
		end
		return out
	end

	-- Convenience accessors (no model file I/O).
	function Model_mt.vocab(self)
		return llama.llama_model_get_vocab(self.model)
	end

	function Model_mt.n_params(self)
		return llama.llama_model_n_params(self.model)
	end

	function Model_mt.desc(self)
		local buf = ffi.new("char[256]")
		local n = llama.llama_model_desc(self.model, buf, 256)
		if n <= 0 then return "" end
		return ffi.string(buf)
	end

	return function(backend, path)
		local mparams = llama.llama_model_default_params()
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
