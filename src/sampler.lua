-- Sampler: wraps llama_sampler_chain (temp + dist, optional top_p). __gc frees chain.

return function(llama)
	local Sampler_mt = {
		__gc = function(self)
			if self.chain ~= nil then
				llama.llama_sampler_free(self.chain)
				self.chain = nil
			end
		end,
		__index = {},
	}
	Sampler_mt.__index = Sampler_mt

	-- Accept a token (e.g. prompt token or previously sampled token) into the sampler.
	function Sampler_mt.accept(self, token)
		llama.llama_sampler_accept(self.chain, token)
	end

	-- Sample next token from context logits at the given index. ctx is raw llama_context* (cdata).
	function Sampler_mt.sample(self, ctx_cdata, logits_idx)
		return llama.llama_sampler_sample(self.chain, ctx_cdata, logits_idx)
	end

	-- Reset sampler state (e.g. for a new sequence).
	function Sampler_mt.reset(self)
		llama.llama_sampler_reset(self.chain)
	end

	-- opts: temp (number), seed (number), top_p (number, optional), min_keep (number, optional for top_p), greedy (bool)
	return function(opts)
		opts = opts or {}
		local temp = opts.temp or 0.7
		local seed = opts.seed or 12345
		local chain_params = llama.llama_sampler_chain_default_params()
		local chain = llama.llama_sampler_chain_init(chain_params)
		if opts.greedy then
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_greedy())
		else
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_temp(temp))
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_dist(seed))
		end
		if opts.top_p and opts.top_p < 1.0 then
			local min_keep = opts.min_keep or 1
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_top_p(opts.top_p, min_keep))
		end
		return setmetatable({
			chain = chain,
		}, Sampler_mt)
	end
end
