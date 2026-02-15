-- Unit tests for lluama.Sampler (no model required; only checks constructor and methods exist)

package.path = "./?.lua;" .. package.path
local lluama = require("src.lluama")

assert(lluama.Sampler, "Sampler constructor exists")
local s = lluama.Sampler({ temp = 0.5, seed = 42 })
assert(s.accept, "accept method")
assert(s.sample, "sample method")
assert(s.reset, "reset method")
assert(s.chain, "chain cdata")
s = nil
collectgarbage("collect")

print("sampler_spec: all passed")
