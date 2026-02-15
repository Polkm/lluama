-- Context: holds model (keeps it alive), .ctx (cdata). __gc frees context.
-- Supports decode_tokens, decode_one (single token), logits, set_sampler.

local ffi = require("ffi")

return function(llama)
	local LLAMA_SEQ_ID = 0
	local seq_id_one = ffi.new("int32_t[1]", LLAMA_SEQ_ID)

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

	-- token_ids: array of token ids. pos_start: optional (default 0) first position in context.
	function Context_mt.decode_tokens(self, token_ids, pos_start)
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

	-- Decode a single token at position pos. Uses an internal one-token batch (no free).
	function Context_mt.decode_one(self, token, pos)
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
		return llama.llama_get_logits(self.ctx)
	end

	-- Attach a sampler to this context for the given seq_id (default 0).
	-- sampler: lluama.Sampler instance (use .chain), or raw llama_sampler* cdata.
	function Context_mt.set_sampler(self, sampler, seq_id)
		seq_id = seq_id or LLAMA_SEQ_ID
		local chain = sampler.chain or sampler
		return llama.llama_set_sampler(self.ctx, seq_id, chain)
	end

	return function(model, ctx)
		return setmetatable({
			ctx = ctx,
			model = model,
		}, Context_mt)
	end
end
