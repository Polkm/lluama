-- Sampler: wraps llama_sampler_chain (temp + dist, optional top_p, optional grammar). __gc frees chain.
-- When opts.grammar is set, pass the model as second argument so the sampler can use the vocab.

return function(llama)
	local grammars = require("src.grammars")
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

	function Sampler_mt.chain_n(self)
		return llama.llama_sampler_chain_n(self.chain)
	end
	function Sampler_mt.chain_get(self, i)
		return llama.llama_sampler_chain_get(self.chain, i)
	end
	function Sampler_mt.get_seed(self)
		return llama.llama_sampler_get_seed(self.chain)
	end

	-- opts: temp, seed, top_p, top_k, min_p, typical, min_keep, greedy, penalty_last_n, penalty_repeat, penalty_freq, penalty_present, grammar, grammar_root
	-- When opts.grammar is set, model (second arg) must be provided for vocab. grammar can be "json" or a GBNF string.
	return function(opts, model)
		opts = opts or {}
		local temp = opts.temp or 0.7
		local seed = opts.seed or 12345
		local chain_params = llama.llama_sampler_chain_default_params()
		local chain = llama.llama_sampler_chain_init(chain_params)
		-- Grammar first so it constrains which tokens are valid before temp/dist.
		local grammar_str = opts.grammar
		if grammar_str then
			if not model then
				error("lluama.Sampler: opts.grammar requires model as second argument")
			end
			local vocab = model:vocab()
			local gstr, root
			if grammar_str == "json" then
				gstr = grammars.json
				root = opts.grammar_root or grammars.json_root
			elseif type(grammar_str) == "string" and grammar_str:find("::=") then
				gstr = grammar_str
				root = opts.grammar_root or "root"
			else
				error("lluama.Sampler: opts.grammar must be \"json\" or a GBNF string containing \"::=\"")
			end
			local grammar_sampler = llama.llama_sampler_init_grammar(vocab, gstr, root)
			if grammar_sampler == nil then
				error("lluama.Sampler: failed to init grammar sampler")
			end
			llama.llama_sampler_chain_add(chain, grammar_sampler)
		end
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
		if opts.top_k and opts.top_k > 0 then
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_top_k(opts.top_k))
		end
		if opts.min_p and opts.min_p > 0 then
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_min_p(opts.min_p, opts.min_keep or 1))
		end
		if opts.typical and opts.typical > 0 then
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_typical(opts.typical, opts.min_keep or 1))
		end
		if opts.penalty_last_n or opts.penalty_repeat or opts.penalty_freq or opts.penalty_present then
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_penalties(
				opts.penalty_last_n or 64,
				opts.penalty_repeat or 1.1,
				opts.penalty_freq or 0.0,
				opts.penalty_present or 0.0
			))
		end
		return setmetatable({
			chain = chain,
		}, Sampler_mt)
	end
end
