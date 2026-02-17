-- Optional FFI for ggml threadpool create/free.
-- Loaded on demand by lluama.threadpool_create. If the build does not export
-- ggml_threadpool_create/ggml_threadpool_free, create and free will be nil.

local ffi = require("ffi")
local bindings = require("src.bindings")
local llama = bindings.llama
local ggml = bindings.ggml

local create, free

local ok = pcall(function()
	ffi.cdef([[
		struct ggml_threadpool {};
		typedef struct ggml_threadpool * ggml_threadpool_t;
		ggml_threadpool_t ggml_threadpool_create(int n_threads);
		void ggml_threadpool_free(ggml_threadpool_t tp);
	]])
	create = llama.ggml_threadpool_create or ggml.ggml_threadpool_create
	free = llama.ggml_threadpool_free or ggml.ggml_threadpool_free
end)

if not ok then
	create = nil
	free = nil
end

return {
	create = create,
	free = free,
}
