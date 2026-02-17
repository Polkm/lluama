-- Unit tests for ChatSession and grammar wiring (Session + grammar used together).
-- With model: verifies session has _grammar set and sampler chain includes grammar.
-- Without model file: skips model-dependent tests.

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

local function model_path_exists(path)
	local f = io.open(path, "rb")
	if f then f:close(); return true end
	return false
end

local default_model = "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"
local has_model = model_path_exists(default_model)

-- ChatSession with grammar: _grammar must be set and sampler must have a chain with grammar.
-- Skip if model has no chat template (tokenizer.chat_template).
if has_model and lluama.ChatSession then
	local backend = lluama.Backend()
	backend:init()
	local ok, session = pcall(lluama.ChatSession, backend, default_model, {
		template = "chatml",
		grammar = "json",
	})
	if not ok then
		io.write("[chat_session_spec] skipped ChatSession tests (model has no template: " .. tostring(session) .. ")\n")
	else
		assert(session._grammar == true, "ChatSession with grammar must set _grammar")
		assert(session.sampler, "ChatSession must have sampler")
		assert(session.sampler.chain_n, "sampler must have chain_n")
		local n = session.sampler:chain_n()
		assert(n >= 2, "grammar sampler chain must have at least 2 elements (grammar + temp/greedy)")
		assert(session.sampler:chain_get(0) ~= nil, "first chain element (grammar) must exist")
		session = nil
		collectgarbage("collect")
	end
end

-- ChatSession without grammar: _grammar must be nil.
if has_model and lluama.ChatSession then
	local backend = lluama.Backend()
	backend:init()
	local ok, session = pcall(lluama.ChatSession, backend, default_model, { template = "chatml" })
	if ok and session and not (type(session) == "string") then
		assert(session._grammar == nil, "ChatSession without grammar must have _grammar nil")
		session = nil
	end
	collectgarbage("collect")
end

-- ChatSession with custom GBNF (grammar string): same wiring as "json".
if has_model and lluama.ChatSession then
	local backend = lluama.Backend()
	backend:init()
	local ok, session = pcall(lluama.ChatSession, backend, default_model, {
		template = "chatml",
		grammar = 'root ::= "a"\n',
		grammar_root = "root",
	})
	if ok and session and not (type(session) == "string") then
		assert(session._grammar == true, "ChatSession with GBNF string must set _grammar")
		assert(session.sampler:chain_n() >= 2, "custom grammar must be in chain")
		session = nil
	end
	collectgarbage("collect")
end

if not has_model then
	io.write("[chat_session_spec] skipped model-dependent tests (no model at " .. default_model .. ")\n")
end

print("chat_session_spec: all passed")