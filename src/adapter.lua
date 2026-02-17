-- LoRA adapter: wrap llama_adapter_lora. Attach/detach via context.
-- No explicit free in the current API; keep the adapter alive while in use and attach to a context when needed.

local ffi = require("ffi")

return function(llama, lluama)
	local Adapter_mt = {
		__index = {},
	}
	Adapter_mt.__index = Adapter_mt

	local function assert_adapter(self)
		if self.adapter == nil then
			error("lluama: adapter already freed or invalid")
		end
	end

	function Adapter_mt.meta_val_str(self, key)
		assert_adapter(self)
		local buf = ffi.new("char[?]", 256)
		local n = llama.llama_adapter_meta_val_str(self.adapter, key, buf, 256)
		return n > 0 and ffi.string(buf, n) or nil
	end

	function Adapter_mt.meta_count(self)
		assert_adapter(self)
		return llama.llama_adapter_meta_count(self.adapter)
	end

	function Adapter_mt.meta_key_by_index(self, i)
		assert_adapter(self)
		local buf = ffi.new("char[?]", 128)
		local n = llama.llama_adapter_meta_key_by_index(self.adapter, i, buf, 128)
		return n > 0 and ffi.string(buf, n) or nil
	end

	function Adapter_mt.meta_val_str_by_index(self, i)
		assert_adapter(self)
		local buf = ffi.new("char[?]", 256)
		local n = llama.llama_adapter_meta_val_str_by_index(self.adapter, i, buf, 256)
		return n > 0 and ffi.string(buf, n) or nil
	end

	function Adapter_mt.alora_n_invocation_tokens(self)
		assert_adapter(self)
		return llama.llama_adapter_get_alora_n_invocation_tokens(self.adapter)
	end

	function Adapter_mt.alora_invocation_tokens(self)
		assert_adapter(self)
		local n = llama.llama_adapter_get_alora_n_invocation_tokens(self.adapter)
		if n == 0 then return {} end
		local p = llama.llama_adapter_get_alora_invocation_tokens(self.adapter)
		if p == nil then return {} end
		local out = {}
		for i = 0, n - 1 do
			out[i + 1] = p[i]
		end
		return out
	end

	-- model: Model instance (or raw llama_model*). path_lora: path to LoRA file.
	local function init(model, path_lora)
		local model_cdata = model.model or model
		if model_cdata == nil then
			error("lluama.AdapterLora: model required")
		end
		local adapter = llama.llama_adapter_lora_init(model_cdata, path_lora)
		if adapter == nil then
			error("lluama: failed to load LoRA adapter: " .. tostring(path_lora))
		end
		return setmetatable({
			adapter = adapter,
			model = model,
		}, Adapter_mt)
	end

	return init
end
