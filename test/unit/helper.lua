-- Shared test scaffolding for unit specs. Load via require("test.unit.helper") when run from repo root.
-- The unit runner (run.lua) sets package.path so this module and src.lluama are findable.

local lluama = require("src.lluama")

local default_model_path = "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"

local function model_path_exists(path)
	local f = io.open(path, "rb")
	if f then
		f:close()
		return true
	end
	return false
end

return {
	lluama = lluama,
	default_model_path = default_model_path,
	model_path_exists = model_path_exists,
}
