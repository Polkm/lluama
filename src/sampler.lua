-- Sampler: wraps llama_sampler_chain (temp + dist, optional top_p, optional grammar). __gc frees chain.
-- When opts.grammar is set, pass the model as second argument so the sampler can use the vocab.

local ffi = require("ffi")

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

	function Sampler_mt.perf_sampler(self)
		local d = llama.llama_perf_sampler(self.chain)
		return { t_sample_ms = d.t_sample_ms, n_sample = d.n_sample }
	end
	function Sampler_mt.perf_sampler_print(self)
		llama.llama_perf_sampler_print(self.chain)
	end
	function Sampler_mt.perf_sampler_reset(self)
		llama.llama_perf_sampler_reset(self.chain)
	end

	function Sampler_mt.clone(self)
		local c = llama.llama_sampler_clone(self.chain)
		if c == nil then return nil end
		return setmetatable({ chain = c }, Sampler_mt)
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
		if opts.temp_ext and type(opts.temp_ext) == "table" then
			local t, delta, exp = opts.temp_ext[1], opts.temp_ext[2], opts.temp_ext[3]
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_temp_ext(t or 0.7, delta or 0, exp or 1))
		end
		if opts.xtc and type(opts.xtc) == "table" then
			local p, t, mk, s = opts.xtc[1], opts.xtc[2], opts.xtc[3] or 1, opts.xtc[4] or seed
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_xtc(p or 0.1, t or 0.7, mk, s))
		end
		if opts.top_n_sigma and opts.top_n_sigma > 0 then
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_top_n_sigma(opts.top_n_sigma))
		end
		if opts.mirostat and type(opts.mirostat) == "table" then
			local n_vocab = opts.mirostat.n_vocab or (model and model:vocab_n_tokens() or 32000)
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_mirostat(
				n_vocab, seed,
				opts.mirostat.tau or 5, opts.mirostat.eta or 0.1, opts.mirostat.m or 0
			))
		end
		if opts.mirostat_v2 then
			local t = type(opts.mirostat_v2) == "table" and opts.mirostat_v2 or {}
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_mirostat_v2(
				seed, t.tau or 5, t.eta or 0.1
			))
		end
		if opts.grammar_lazy and model then
			local g = opts.grammar_lazy
			local vocab = model:vocab()
			local gstr = g.grammar_str or g[1]
			local root = g.grammar_root or g.root or "root"
			local trigger_patterns = g.trigger_patterns or {}
			local trigger_tokens = g.trigger_tokens or {}
			local n_pat = #trigger_patterns
			local n_tok = #trigger_tokens
			local ptr_ptrs = n_pat > 0 and ffi.new("const char*[?]", n_pat) or nil
			if ptr_ptrs then for i = 0, n_pat - 1 do ptr_ptrs[i] = trigger_patterns[i + 1] end end
			local tok_buf = n_tok > 0 and ffi.new("int32_t[?]", n_tok) or nil
			if tok_buf then for i = 0, n_tok - 1 do tok_buf[i] = trigger_tokens[i + 1] end end
			local smpl = llama.llama_sampler_init_grammar_lazy_patterns(vocab, gstr, root, ptr_ptrs, n_pat, tok_buf, n_tok)
			if smpl then llama.llama_sampler_chain_add(chain, smpl) end
		end
		if opts.dry and type(opts.dry) == "table" and model then
			local d = opts.dry
			local n_ctx_train = d.n_ctx_train or model:n_ctx_train()
			local breakers = d.seq_breakers or {}
			local n_br = #breakers
			local br_ptrs = n_br > 0 and ffi.new("const char*[?]", n_br) or nil
			if br_ptrs then for i = 0, n_br - 1 do br_ptrs[i] = breakers[i + 1] end end
			local smpl = llama.llama_sampler_init_dry(model:vocab(), n_ctx_train,
				d.dry_multiplier or 1, d.dry_base or 0, d.dry_allowed_length or 0, d.dry_penalty_last_n or 64,
				br_ptrs, n_br)
			if smpl then llama.llama_sampler_chain_add(chain, smpl) end
		end
		if opts.adaptive_p and type(opts.adaptive_p) == "table" then
			local a = opts.adaptive_p
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_adaptive_p(a.target or 0.5, a.decay or 0.99, seed))
		end
		if opts.logit_bias and type(opts.logit_bias) == "table" and #opts.logit_bias > 0 then
			local n_vocab = model and model:vocab_n_tokens() or 32000
			local n_lb = #opts.logit_bias
			local buf = ffi.new("llama_logit_bias[?]", n_lb)
			for i = 0, n_lb - 1 do
				local e = opts.logit_bias[i + 1]
				buf[i].token = e.token or e[1]
				buf[i].bias = e.bias or e[2] or 0
			end
			llama.llama_sampler_chain_add(chain, llama.llama_sampler_init_logit_bias(n_vocab, n_lb, buf))
		end
		if opts.infill and model then
			local smpl = llama.llama_sampler_init_infill(model:vocab())
			if smpl then llama.llama_sampler_chain_add(chain, smpl) end
		end
		return setmetatable({
			chain = chain,
		}, Sampler_mt)
	end
end
