-- Context: holds model (keeps it alive), .ctx (cdata). __gc frees context.
-- Supports decode_tokens, decode_one, encode, logits, set_sampler, state, etc.

local ffi = require("ffi")

return function(llama)
	local LLAMA_SEQ_ID = 0
	local seq_id_one = ffi.new("int32_t[1]", LLAMA_SEQ_ID)

	local function assert_ctx(self)
		if self.ctx == nil then
			error("lluama: context already freed")
		end
	end

	local Context_mt = {
		__gc = function(self)
			if self.ctx ~= nil then
				llama.llama_free(self.ctx)
				self.ctx = nil
			end
		end,
		__index = {},
	}
	Context_mt.__index = Context_mt

	-- Query
	function Context_mt.n_ctx(self)
		assert_ctx(self)
		return llama.llama_n_ctx(self.ctx)
	end
	function Context_mt.n_ctx_seq(self)
		assert_ctx(self)
		return llama.llama_n_ctx_seq(self.ctx)
	end
	function Context_mt.n_batch(self)
		assert_ctx(self)
		return llama.llama_n_batch(self.ctx)
	end
	function Context_mt.n_ubatch(self)
		assert_ctx(self)
		return llama.llama_n_ubatch(self.ctx)
	end
	function Context_mt.n_seq_max(self)
		assert_ctx(self)
		return llama.llama_n_seq_max(self.ctx)
	end
	function Context_mt.get_model(self)
		assert_ctx(self)
		return llama.llama_get_model(self.ctx)
	end
	function Context_mt.pooling_type(self)
		assert_ctx(self)
		return llama.llama_pooling_type(self.ctx)
	end

	-- Threads
	function Context_mt.set_n_threads(self, n_threads, n_threads_batch)
		assert_ctx(self)
		llama.llama_set_n_threads(self.ctx, n_threads or -1, n_threads_batch or -1)
	end
	function Context_mt.n_threads(self)
		assert_ctx(self)
		return llama.llama_n_threads(self.ctx)
	end
	function Context_mt.n_threads_batch(self)
		assert_ctx(self)
		return llama.llama_n_threads_batch(self.ctx)
	end

	-- Embeddings / eval options
	function Context_mt.set_embeddings(self, embeddings)
		assert_ctx(self)
		llama.llama_set_embeddings(self.ctx, embeddings)
	end
	function Context_mt.set_causal_attn(self, causal_attn)
		assert_ctx(self)
		llama.llama_set_causal_attn(self.ctx, causal_attn)
	end
	function Context_mt.set_warmup(self, warmup)
		assert_ctx(self)
		llama.llama_set_warmup(self.ctx, warmup)
	end
	function Context_mt.synchronize(self)
		assert_ctx(self)
		llama.llama_synchronize(self.ctx)
	end

	-- token_ids: array of token ids. pos_start: optional (default 0) first position in context.
	function Context_mt.decode_tokens(self, token_ids, pos_start)
		assert_ctx(self)
		local n = #token_ids
		if n == 0 then return 0 end
		pos_start = pos_start or 0
		-- Use a batch with explicit seq_id (0) and logits only on last position.
		local batch = self._batch
		if not batch or self._batch_cap < n then
			self._batch_cap = math.max(n, 256)
			batch = llama.llama_batch_init(self._batch_cap, 0, 1)
			self._batch = batch
		end
		for i = 0, n - 1 do
			batch.token[i] = token_ids[i + 1]
			batch.pos[i] = pos_start + i
			batch.n_seq_id[i] = 1
			batch.seq_id[i] = seq_id_one
			batch.logits[i] = (i == n - 1) and 1 or 0
		end
		batch.n_tokens = n
		return llama.llama_decode(self.ctx, batch)
	end

	-- Encode batch (e.g. for encoder models). batch: llama_batch cdata.
	function Context_mt.encode(self, batch)
		assert_ctx(self)
		return llama.llama_encode(self.ctx, batch)
	end

	-- Decode a single token at position pos. Uses an internal one-token batch (no free).
	function Context_mt.decode_one(self, token, pos)
		assert_ctx(self)
		local batch = self._one_batch
		if not batch then
			batch = llama.llama_batch_init(1, 0, 1)
			self._one_batch = batch
		end
		batch.n_tokens = 1
		batch.token[0] = token
		batch.pos[0] = pos
		batch.n_seq_id[0] = 1
		batch.seq_id[0] = seq_id_one
		batch.logits[0] = 1
		return llama.llama_decode(self.ctx, batch)
	end

	function Context_mt.logits(self)
		assert_ctx(self)
		return llama.llama_get_logits(self.ctx)
	end
	function Context_mt.logits_ith(self, i)
		assert_ctx(self)
		return llama.llama_get_logits_ith(self.ctx, i)
	end
	function Context_mt.embeddings(self)
		assert_ctx(self)
		return llama.llama_get_embeddings(self.ctx)
	end
	function Context_mt.embeddings_ith(self, i)
		assert_ctx(self)
		return llama.llama_get_embeddings_ith(self.ctx, i)
	end
	function Context_mt.embeddings_seq(self, seq_id)
		assert_ctx(self)
		return llama.llama_get_embeddings_seq(self.ctx, seq_id or LLAMA_SEQ_ID)
	end

	-- Attach a sampler to this context for the given seq_id (default 0).
	-- sampler: lluama.Sampler instance (use .chain), or raw llama_sampler* cdata.
	function Context_mt.set_sampler(self, sampler, seq_id)
		assert_ctx(self)
		seq_id = seq_id or LLAMA_SEQ_ID
		local chain = sampler.chain or sampler
		return llama.llama_set_sampler(self.ctx, seq_id, chain)
	end

	-- State save/load
	function Context_mt.state_get_size(self)
		assert_ctx(self)
		return llama.llama_state_get_size(self.ctx)
	end
	function Context_mt.state_get_data(self, dst_buf_or_nil)
		assert_ctx(self)
		local sz = llama.llama_state_get_size(self.ctx)
		local buf = dst_buf_or_nil or ffi.new("uint8_t[?]", sz)
		local n = llama.llama_state_get_data(self.ctx, buf, sz)
		return buf, n
	end
	function Context_mt.state_set_data(self, src_buf, size)
		assert_ctx(self)
		return llama.llama_state_set_data(self.ctx, src_buf, size or ffi.sizeof(src_buf))
	end
	-- token_ids: Lua array of tokens to save with the KV state.
	function Context_mt.state_save_file(self, path_session, token_ids)
		assert_ctx(self)
		local n = token_ids and #token_ids or 0
		local tokens_ptr = nil
		if n > 0 then
			local buf = ffi.new("int32_t[?]", n)
			for i = 0, n - 1 do buf[i] = token_ids[i + 1] end
			tokens_ptr = buf
		end
		return llama.llama_state_save_file(self.ctx, path_session, tokens_ptr, n)
	end
	-- Returns Lua array of tokens loaded from file, or nil on failure.
	function Context_mt.state_load_file(self, path_session, n_cap)
		assert_ctx(self)
		local cap = n_cap or 65536
		local token_buffer = ffi.new("int32_t[?]", cap)
		local n_out = ffi.new("size_t[1]")
		local ok = llama.llama_state_load_file(self.ctx, path_session, token_buffer, cap, n_out)
		if not ok then return nil end
		local out = {}
		for i = 0, n_out[0] - 1 do out[i + 1] = token_buffer[i] end
		return out
	end

	-- Sequence state (per seq_id)
	function Context_mt.state_seq_get_size(self, seq_id)
		assert_ctx(self)
		return llama.llama_state_seq_get_size(self.ctx, seq_id or LLAMA_SEQ_ID)
	end
	function Context_mt.state_seq_get_data(self, dst_buf, size, seq_id)
		assert_ctx(self)
		return llama.llama_state_seq_get_data(self.ctx, dst_buf, size, seq_id or LLAMA_SEQ_ID)
	end
	function Context_mt.state_seq_set_data(self, src_buf, size, dest_seq_id)
		assert_ctx(self)
		return llama.llama_state_seq_set_data(self.ctx, src_buf, size or ffi.sizeof(src_buf), dest_seq_id or LLAMA_SEQ_ID)
	end

	return function(model, ctx)
		return setmetatable({
			ctx = ctx,
			model = model,
		}, Context_mt)
	end
end
