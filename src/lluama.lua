-- lluama: Lua bindings for llama.cpp
-- Main module. All llama_* and ggml_* live under .llama and .ggml.
-- Classes: Backend, Model, Context (from src/backend.lua, src/model.lua, src/context.lua).

local ffi = require("ffi")
local bindings = require("src.bindings")
local grammars = require("src.grammars")

local llama = bindings.llama
local ggml = bindings.ggml
local chat_native = require("src.chat_native")(llama)
local lluama = {
	llama = llama,
	ggml = ggml,
	chat_native = chat_native,
	grammars = grammars,
}

-- Logging: silent by default. Use set_log_callback to hook in.
local log_handler = nil
local function log_trampoline(level, text, user_data)
	if log_handler then
		log_handler(level, text and ffi.string(text) or "")
	end
end
local log_trampoline_cb = ffi.cast("llama_log_callback", log_trampoline)
local log_noop_cb = ffi.cast("llama_log_callback", function(level, text, user_data) end)

function lluama.set_log_callback(handler)
	log_handler = handler
	llama.llama_log_set(handler and log_trampoline_cb or log_noop_cb, nil)
end

function lluama.get_log_callback()
	return log_handler
end

-- Silent by default
lluama.set_log_callback(nil)

-- Low-level API parity: time, system info, capabilities
function lluama.time_us()
	return llama.llama_time_us()
end
function lluama.print_system_info()
	local p = llama.llama_print_system_info()
	return p ~= nil and ffi.string(p) or nil
end
-- type_enum: -1 = auto, 0 = disabled, 1 = enabled (llama_flash_attn_type)
function lluama.flash_attn_type_name(type_enum)
	local p = llama.llama_flash_attn_type_name(type_enum)
	return p ~= nil and ffi.string(p) or nil
end
function lluama.supports_mmap() return llama.llama_supports_mmap() end
function lluama.supports_mlock() return llama.llama_supports_mlock() end
function lluama.supports_gpu_offload() return llama.llama_supports_gpu_offload() end
function lluama.supports_rpc() return llama.llama_supports_rpc() end
function lluama.max_devices() return llama.llama_max_devices() end
function lluama.max_parallel_sequences() return llama.llama_max_parallel_sequences() end
function lluama.max_tensor_buft_overrides() return llama.llama_max_tensor_buft_overrides() end

function lluama.chat_apply_template(tmpl, messages, add_ass)
	return chat_native.chat_apply_template(tmpl, messages, add_ass)
end
function lluama.chat_builtin_templates()
	return chat_native.chat_builtin_templates()
end

-- Split model paths: build path for split N of M; get prefix from split path. Return string or nil on error.
function lluama.split_path(path_prefix, split_no, split_count)
	local buf = ffi.new("char[?]", 1024)
	local r = llama.llama_split_path(buf, 1024, path_prefix, split_no, split_count)
	if r ~= 0 then return nil end
	return ffi.string(buf)
end
function lluama.split_prefix(split_path, split_no, split_count)
	local buf = ffi.new("char[?]", 1024)
	local r = llama.llama_split_prefix(buf, 1024, split_path, split_no, split_count)
	if r ~= 0 then return nil end
	return ffi.string(buf)
end

-- Model meta key enum to string (llama_model_meta_key: 0=sampling_sequence, 1=top_k, ...).
function lluama.model_meta_key_str(key_enum)
	local p = llama.llama_model_meta_key_str(key_enum)
	return p ~= nil and ffi.string(p) or nil
end

-- Fit model/context params to a model file. opts: n_ctx_min (default 0), log_level (default 0). Returns status: 0=success, 1=failure, 2=error.
function lluama.params_fit(path_model, opts)
	opts = opts or {}
	local mparams = ffi.new("llama_model_params[1]")
	mparams[0] = llama.llama_model_default_params()
	local cparams = ffi.new("llama_context_params[1]")
	cparams[0] = llama.llama_context_default_params()
	local status = llama.llama_params_fit(
		path_model,
		mparams,
		cparams,
		nil,
		nil,
		nil,
		opts.n_ctx_min or 0,
		opts.log_level or 0
	)
	return status
end

-- Quantize model file. opts: nthread, ftype (enum number), etc. Returns 0 on success.
function lluama.model_quantize(path_in, path_out, opts)
	opts = opts or {}
	local qparams = llama.llama_model_quantize_default_params()
	if opts.nthread ~= nil then qparams.nthread = opts.nthread end
	if opts.ftype ~= nil then qparams.ftype = opts.ftype end
	return llama.llama_model_quantize(path_in, path_out, qparams)
end

-- Build a single-sequence batch from token array. Returns (batch_cdata, token_buf). Caller must llama.llama_batch_free(batch) when done.
function lluama.batch_get_one(token_ids)
	local n = #token_ids
	if n == 0 then
		local batch = llama.llama_batch_get_one(nil, 0)
		return batch, nil
	end
	local buf = ffi.new("int32_t[?]", n)
	for i = 0, n - 1 do buf[i] = token_ids[i + 1] end
	local batch = llama.llama_batch_get_one(buf, n)
	return batch, buf
end

-- Load class constructors (Context before Model so Model can use lluama.Context)
lluama.Backend = require("src.backend")(llama, ggml)
lluama.Context = require("src.context")(llama)
local model_load, model_load_from_splits = require("src.model")(llama, lluama)
lluama.Model = model_load
lluama.load_model_from_splits = model_load_from_splits
lluama.Sampler = require("src.sampler")(llama)
lluama.AdapterLora = require("src.adapter")(llama, lluama)
lluama.ChatSession = require("src.chat_session")(lluama)

return lluama
