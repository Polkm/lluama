-- lluama: Lua bindings for llama.cpp
-- Main module. All llama_* and ggml_* live under .llama and .ggml.
-- Classes: Backend, Model, Context (from src/backend.lua, src/model.lua, src/context.lua).

local ffi = require("ffi")
local bindings = require("src.bindings")
local chat_templates = require("src.chat_templates")

local llama = bindings.llama
local ggml = bindings.ggml
local lluama = {
	llama = llama,
	ggml = ggml,
	chat_templates = chat_templates,
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
function lluama.supports_mmap() return llama.llama_supports_mmap() end
function lluama.supports_mlock() return llama.llama_supports_mlock() end
function lluama.supports_gpu_offload() return llama.llama_supports_gpu_offload() end
function lluama.supports_rpc() return llama.llama_supports_rpc() end
function lluama.max_devices() return llama.llama_max_devices() end
function lluama.max_parallel_sequences() return llama.llama_max_parallel_sequences() end
function lluama.max_tensor_buft_overrides() return llama.llama_max_tensor_buft_overrides() end

-- Load class constructors (Context before Model so Model can use lluama.Context)
lluama.Backend = require("src.backend")(llama, ggml)
lluama.Context = require("src.context")(llama)
lluama.Model = require("src.model")(llama, lluama)
lluama.Sampler = require("src.sampler")(llama)
lluama.ChatSession = require("src.chat_session")(lluama)

return lluama
