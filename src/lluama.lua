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

-- Load class constructors (Context before Model so Model can use lluama.Context)
lluama.Backend = require("src.backend")(llama, ggml)
lluama.Context = require("src.context")(llama)
lluama.Model = require("src.model")(llama, lluama)
lluama.Sampler = require("src.sampler")(llama)
lluama.ChatSession = require("src.chat_session")(lluama)

return lluama
