-- Context: holds model (keeps it alive), .ctx (cdata). __gc frees context.
-- Supports decode_tokens, decode_one, encode, logits, set_sampler, state, etc.

local ffi = require("ffi")

return function(llama)
	local LLAMA_SEQ_ID = 0
	local seq_id_one = ffi.new("int32_t[1]", LLAMA_SEQ_ID)
	local context_abort_registry = {}
	local abort_cb_type = ffi.typeof("bool (*)(void*)")
	local abort_cb = ffi.cast(abort_cb_type, function(ud)
		if ud == nil then return false end
		local ctx = ffi.cast("struct llama_context*", ud)
		local fn = context_abort_registry[ctx]
		return fn and fn() or false
	end)

	local function assert_ctx(self)
		if self.ctx == nil then
			error("lluama: context already freed")
		end
	end

	local Context_mt = {
		__gc = function(self)
			if self._batch ~= nil then
				llama.llama_batch_free(self._batch)
				self._batch = nil
			end
			if self._one_batch ~= nil then
				llama.llama_batch_free(self._one_batch)
				self._one_batch = nil
			end
			if self.ctx ~= nil then
				context_abort_registry[self.ctx] = nil
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
	-- Set abort callback for long evals. cb() should return true to abort. Pass nil to clear.
	function Context_mt.set_abort_callback(self, cb)
		assert_ctx(self)
		if cb == nil then
			context_abort_registry[self.ctx] = nil
			llama.llama_set_abort_callback(self.ctx, nil, nil)
		else
			context_abort_registry[self.ctx] = cb
			llama.llama_set_abort_callback(self.ctx, abort_cb, self.ctx)
		end
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
			if self._batch ~= nil then
				llama.llama_batch_free(self._batch)
			end
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
	-- Top-k token ids and probs from logits at position i (softmax). Works with client-side sampling.
	-- Keeps only top-k in a single pass to avoid O(n_vocab) allocations and full sort.
	function Context_mt.logits_top_k_ith(self, i, k)
		assert_ctx(self)
		k = k or 10
		local logits = llama.llama_get_logits_ith(self.ctx, i)
		if logits == nil then return {} end
		local model = llama.llama_get_model(self.ctx)
		if model == nil then return {} end
		local n_vocab = llama.llama_vocab_n_tokens(llama.llama_model_get_vocab(model))
		if n_vocab <= 0 then return {} end
		local max_logit = -1e38
		for j = 0, n_vocab - 1 do
			if logits[j] > max_logit then max_logit = logits[j] end
		end
		local sum_exp = 0
		for j = 0, n_vocab - 1 do
			sum_exp = sum_exp + math.exp(logits[j] - max_logit)
		end
		if sum_exp <= 0 then return {} end
		local top = {}
		for j = 0, n_vocab - 1 do
			local p = math.exp(logits[j] - max_logit) / sum_exp
			if #top < k then
				local idx = #top + 1
				for ii = 1, #top do
					if p > top[ii].p then idx = ii; break end
				end
				table.insert(top, idx, { id = j, p = p })
			elseif p > top[k].p then
				local idx = k
				for ii = 1, k do
					if p > top[ii].p then idx = ii; break end
				end
				for ii = k, idx + 1, -1 do top[ii] = top[ii - 1] end
				top[idx] = { id = j, p = p }
			end
		end
		return top
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
	-- Save full context state (rng, logits, embedding, kv_cache) to a binary file.
	-- Mirrors save-load-state example: state_get_size -> state_get_data -> write file.
	-- Returns bytes_written, or nil, errmsg on failure.
	function Context_mt.state_save_blob(self, path)
		assert_ctx(self)
		local buf, n = self:state_get_data(nil)
		if n == 0 then return 0 end
		local f = io.open(path, "wb")
		if not f then return nil, "state_save_blob: failed to open " .. tostring(path) end
		local ok, err
		ok, err = pcall(function()
			f:write(ffi.string(buf, n))
		end)
		f:close()
		if not ok then return nil, err or "write failed" end
		return n
	end
	-- Load full context state from a binary file (as saved by state_save_blob).
	-- Returns bytes_read, or nil, errmsg on failure.
	function Context_mt.state_load_blob(self, path)
		assert_ctx(self)
		local f = io.open(path, "rb")
		if not f then return nil, "state_load_blob: failed to open " .. tostring(path) end
		local data = f:read("*a")
		f:close()
		if not data or #data == 0 then return nil, "state_load_blob: empty file" end
		local buf = ffi.new("uint8_t[?]", #data)
		ffi.copy(buf, data, #data)
		local nset = llama.llama_state_set_data(self.ctx, buf, #data)
		if nset ~= #data then return nil, "state_load_blob: set_data length mismatch" end
		return nset
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
	-- Per-sequence state save/load to file
	function Context_mt.state_seq_save_file(self, path, seq_id, token_ids)
		assert_ctx(self)
		seq_id = seq_id or LLAMA_SEQ_ID
		local n = token_ids and #token_ids or 0
		local tokens_ptr = nil
		if n > 0 then
			local buf = ffi.new("int32_t[?]", n)
			for i = 0, n - 1 do buf[i] = token_ids[i + 1] end
			tokens_ptr = buf
		end
		return llama.llama_state_seq_save_file(self.ctx, path, seq_id, tokens_ptr, n)
	end
	function Context_mt.state_seq_load_file(self, path, dest_seq_id, n_cap)
		assert_ctx(self)
		local cap = n_cap or 65536
		local token_buffer = ffi.new("int32_t[?]", cap)
		local n_out = ffi.new("size_t[1]")
		local sz = llama.llama_state_seq_load_file(self.ctx, path, dest_seq_id or LLAMA_SEQ_ID, token_buffer, cap, n_out)
		if sz == 0 then return nil end
		local out = {}
		for i = 0, n_out[0] - 1 do out[i + 1] = token_buffer[i] end
		return out
	end
	-- Extended state (flags: uint32_t, pass as number)
	function Context_mt.state_seq_get_size_ext(self, seq_id, flags)
		assert_ctx(self)
		return llama.llama_state_seq_get_size_ext(self.ctx, seq_id or LLAMA_SEQ_ID, flags or 0)
	end
	function Context_mt.state_seq_get_data_ext(self, dst_buf, size, seq_id, flags)
		assert_ctx(self)
		return llama.llama_state_seq_get_data_ext(self.ctx, dst_buf, size, seq_id or LLAMA_SEQ_ID, flags or 0)
	end
	function Context_mt.state_seq_set_data_ext(self, src_buf, size, dest_seq_id, flags)
		assert_ctx(self)
		return llama.llama_state_seq_set_data_ext(self.ctx, src_buf, size or ffi.sizeof(src_buf), dest_seq_id or LLAMA_SEQ_ID, flags or 0)
	end
	-- Copy one sequence's KV state to another (e.g. save seq 0, clear KV, restore into seq 1).
	-- If clear_before then KV cache is cleared before restoring into dest_seq_id.
	-- Returns bytes copied, or nil, errmsg on failure.
	function Context_mt.state_seq_copy(self, src_seq_id, dest_seq_id, clear_before)
		assert_ctx(self)
		src_seq_id = src_seq_id or LLAMA_SEQ_ID
		dest_seq_id = dest_seq_id or LLAMA_SEQ_ID
		local sz = self:state_seq_get_size(src_seq_id)
		if sz == 0 then return 0 end
		local buf = ffi.new("uint8_t[?]", sz)
		local nget = self:state_seq_get_data(buf, sz, src_seq_id)
		if nget ~= sz then return nil, "state_seq_copy: get_data length mismatch" end
		if clear_before then
			self:memory_clear(true)
		end
		local nset = self:state_seq_set_data(buf, sz, dest_seq_id)
		if nset ~= sz then return nil, "state_seq_copy: set_data length mismatch" end
		return nset
	end

	-- Sampled token/probs/candidates (when using a sampler)
	function Context_mt.sampled_token_ith(self, i)
		assert_ctx(self)
		return llama.llama_get_sampled_token_ith(self.ctx, i)
	end
	function Context_mt.sampled_probs_count_ith(self, i)
		assert_ctx(self)
		return llama.llama_get_sampled_probs_count_ith(self.ctx, i)
	end
	function Context_mt.sampled_probs_ith(self, i)
		assert_ctx(self)
		local n = llama.llama_get_sampled_probs_count_ith(self.ctx, i)
		if n == 0 then return {} end
		local p = llama.llama_get_sampled_probs_ith(self.ctx, i)
		if p == nil then return {} end
		local out = {}
		for j = 0, n - 1 do out[j + 1] = p[j] end
		return out
	end
	function Context_mt.sampled_logits_count_ith(self, i)
		assert_ctx(self)
		return llama.llama_get_sampled_logits_count_ith(self.ctx, i)
	end
	function Context_mt.sampled_logits_ith(self, i)
		assert_ctx(self)
		local n = llama.llama_get_sampled_logits_count_ith(self.ctx, i)
		if n == 0 then return {} end
		local p = llama.llama_get_sampled_logits_ith(self.ctx, i)
		if p == nil then return {} end
		local out = {}
		for j = 0, n - 1 do out[j + 1] = p[j] end
		return out
	end
	function Context_mt.sampled_candidates_count_ith(self, i)
		assert_ctx(self)
		return llama.llama_get_sampled_candidates_count_ith(self.ctx, i)
	end
	function Context_mt.sampled_candidates_ith(self, i)
		assert_ctx(self)
		local n = llama.llama_get_sampled_candidates_count_ith(self.ctx, i)
		if n == 0 then return {} end
		local p = llama.llama_get_sampled_candidates_ith(self.ctx, i)
		if p == nil then return {} end
		local out = {}
		for j = 0, n - 1 do out[j + 1] = p[j] end
		return out
	end
	-- Top-k sampled token ids and probs for the ith output. Returns { { id = token_id, p = prob }, ... }.
	function Context_mt.sampled_top_k_ith(self, i, k)
		assert_ctx(self)
		k = k or 10
		local nc = llama.llama_get_sampled_candidates_count_ith(self.ctx, i)
		local np = llama.llama_get_sampled_probs_count_ith(self.ctx, i)
		if nc == 0 or np == 0 then return {} end
		local cand = llama.llama_get_sampled_candidates_ith(self.ctx, i)
		local probs = llama.llama_get_sampled_probs_ith(self.ctx, i)
		if cand == nil or probs == nil then return {} end
		local n = math.min(nc, np)
		local list = {}
		for j = 0, n - 1 do
			list[j + 1] = { id = cand[j], p = probs[j] }
		end
		table.sort(list, function(a, b) return a.p > b.p end)
		local out = {}
		for j = 1, math.min(k, #list) do out[j] = list[j] end
		return out
	end

	-- Performance
	function Context_mt.perf_context(self)
		assert_ctx(self)
		local d = llama.llama_perf_context(self.ctx)
		return {
			t_start_ms = d.t_start_ms,
			t_load_ms = d.t_load_ms,
			t_p_eval_ms = d.t_p_eval_ms,
			t_eval_ms = d.t_eval_ms,
			n_p_eval = d.n_p_eval,
			n_eval = d.n_eval,
			n_reused = d.n_reused,
		}
	end
	function Context_mt.perf_context_print(self)
		assert_ctx(self)
		llama.llama_perf_context_print(self.ctx)
	end
	function Context_mt.perf_context_reset(self)
		assert_ctx(self)
		llama.llama_perf_context_reset(self.ctx)
	end
	function Context_mt.memory_breakdown_print(self)
		assert_ctx(self)
		llama.llama_memory_breakdown_print(self.ctx)
	end

	-- KV memory handle (opaque; for memory_* APIs)
	function Context_mt.get_memory(self)
		assert_ctx(self)
		return llama.llama_get_memory(self.ctx)
	end
	-- KV memory operations (multi-sequence)
	function Context_mt.memory_clear(self, data)
		assert_ctx(self)
		llama.llama_memory_clear(llama.llama_get_memory(self.ctx), data == true)
	end
	function Context_mt.memory_seq_rm(self, seq_id, p0, p1)
		assert_ctx(self)
		return llama.llama_memory_seq_rm(llama.llama_get_memory(self.ctx), seq_id, p0, p1)
	end
	function Context_mt.memory_seq_cp(self, seq_id_src, seq_id_dst, p0, p1)
		assert_ctx(self)
		llama.llama_memory_seq_cp(llama.llama_get_memory(self.ctx), seq_id_src, seq_id_dst, p0, p1)
	end
	function Context_mt.memory_seq_keep(self, seq_id)
		assert_ctx(self)
		llama.llama_memory_seq_keep(llama.llama_get_memory(self.ctx), seq_id)
	end
	function Context_mt.memory_seq_add(self, seq_id, p0, p1, delta)
		assert_ctx(self)
		llama.llama_memory_seq_add(llama.llama_get_memory(self.ctx), seq_id, p0, p1, delta)
	end
	function Context_mt.memory_seq_div(self, seq_id, p0, p1, d)
		assert_ctx(self)
		llama.llama_memory_seq_div(llama.llama_get_memory(self.ctx), seq_id, p0, p1, d)
	end
	function Context_mt.memory_seq_pos_min(self, seq_id)
		assert_ctx(self)
		return llama.llama_memory_seq_pos_min(llama.llama_get_memory(self.ctx), seq_id)
	end
	function Context_mt.memory_seq_pos_max(self, seq_id)
		assert_ctx(self)
		return llama.llama_memory_seq_pos_max(llama.llama_get_memory(self.ctx), seq_id)
	end
	function Context_mt.memory_can_shift(self)
		assert_ctx(self)
		return llama.llama_memory_can_shift(llama.llama_get_memory(self.ctx))
	end

	-- Threadpool (advanced; threadpool_cdata is ggml type)
	function Context_mt.attach_threadpool(self, threadpool, threadpool_batch)
		assert_ctx(self)
		llama.llama_attach_threadpool(self.ctx, threadpool, threadpool_batch)
	end
	function Context_mt.detach_threadpool(self)
		assert_ctx(self)
		llama.llama_detach_threadpool(self.ctx)
	end

	-- LoRA adapter
	function Context_mt.set_adapter_lora(self, adapter, scale)
		assert_ctx(self)
		local adptr = adapter.adapter or adapter
		return llama.llama_set_adapter_lora(self.ctx, adptr, scale or 1.0)
	end
	function Context_mt.rm_adapter_lora(self, adapter)
		assert_ctx(self)
		local adptr = adapter.adapter or adapter
		return llama.llama_rm_adapter_lora(self.ctx, adptr)
	end
	function Context_mt.clear_adapter_lora(self)
		assert_ctx(self)
		llama.llama_clear_adapter_lora(self.ctx)
	end
	function Context_mt.apply_adapter_cvec(self, data, len, n_embd, il_start, il_end)
		assert_ctx(self)
		local buf
		if type(data) == "table" then
			len = len or #data
			buf = ffi.new("float[?]", len)
			for i = 0, len - 1 do buf[i] = data[i + 1] end
		else
			buf = data
			len = len or ffi.sizeof(buf) / ffi.sizeof("float")
		end
		return llama.llama_apply_adapter_cvec(self.ctx, buf, len, n_embd, il_start, il_end)
	end

	return function(model, ctx)
		return setmetatable({
			ctx = ctx,
			model = model,
		}, Context_mt)
	end
end
