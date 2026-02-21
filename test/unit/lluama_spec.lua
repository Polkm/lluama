-- Unit tests for lluama module and class constructors (no model file required)

local helper = require("test.unit.helper")
local lluama = helper.lluama

assert(lluama.llama, "lluama.llama")
assert(lluama.ggml, "lluama.ggml")
assert(lluama.grammars and lluama.grammars.json and lluama.grammars.json_root == "root", "lluama.grammars.json")
assert(type(lluama.Backend) == "function", "Backend is constructor")
assert(type(lluama.Model) == "function", "Model is constructor")
assert(type(lluama.Context) == "function", "Context is constructor")
assert(lluama.set_log_callback, "set_log_callback exists")
-- Sampler if present
if lluama.Sampler then
	assert(type(lluama.Sampler) == "function", "Sampler is constructor")
end
if lluama.ChatSession then
	assert(type(lluama.ChatSession) == "function", "ChatSession is constructor")
end

print("lluama_spec: all passed")
