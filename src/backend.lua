-- Backend: process-wide backend. Keep alive while using any Model/Context.
-- __gc calls llama_backend_free.

return function(llama, ggml)
	local Backend_mt = {
		__gc = function(self)
			llama.llama_backend_free()
		end,
		__index = {},
	}
	Backend_mt.__index = Backend_mt

	function Backend_mt.init(self)
		if not ggml.ggml_backend_load_all() then
			ggml.ggml_backend_load("ggml-cpu-x64")
		end
		llama.llama_backend_init()
	end

	-- numa: ggml_numa_strategy enum (e.g. 0 = disabled). Call before init() if needed.
	function Backend_mt.numa_init(self, numa)
		llama.llama_numa_init(numa or 0)
	end

	return function()
		return setmetatable({}, Backend_mt)
	end
end
