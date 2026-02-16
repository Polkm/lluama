-- Unit tests for lluama.Sampler (no model required for basic tests; grammar tests need model)

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

assert(lluama.Sampler, "Sampler constructor exists")
local s = lluama.Sampler({ temp = 0.5, seed = 42 })
assert(s.accept, "accept method")
assert(s.sample, "sample method")
assert(s.reset, "reset method")
assert(s.chain, "chain cdata")
assert(s.chain_n, "chain_n method")
assert(s.chain_get, "chain_get method")
s = nil
collectgarbage("collect")

-- Grammar requires model as second argument (no model = error).
local ok, err = pcall(lluama.Sampler, { grammar = "json" })
assert(not ok and tostring(err):find("grammar"), "Sampler with grammar and no model must error")

-- Custom GBNF string also requires model.
ok, err = pcall(lluama.Sampler, { grammar = 'root ::= "a"\n', grammar_root = "root" })
assert(not ok and tostring(err):find("grammar"), "Sampler with GBNF and no model must error")

print("sampler_spec: all passed")
