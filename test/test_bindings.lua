-- Quick test: load a GGUF model and run one decode step.
-- Run from repo root: luajit test/test_bindings.lua
--
-- Requires:
--   - LuaJIT (with FFI)
--   - llama.cpp shared lib (and GGML backends) on PATH / library path
--   - A GGUF model at the path in MODEL_PATH below (edit as needed)

local lluama = require("src.lluama")

-- Path to GGUF model, relative to repo root. Edit as needed. Run from repo root.
local MODEL_PATH = "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_S.gguf"

local function main()
    local backend = lluama.Backend()
    backend:init()
    print("Backend initialized")

    local model = lluama.Model(backend, MODEL_PATH)
    local ctx = model:context({ n_ctx = 256, n_batch = 128 })

    local tokens = model:tokenize("Hello")
    local err = ctx:decode_tokens({ tokens[1] })  -- one decode step
    if err ~= 0 then
        error("Decode failed with " .. tostring(err))
    end

    local logits = ctx:logits()
    if logits == nil then
        error("Get logits failed")
    end

    -- No explicit free: Backend, Model, Context have __gc (context → model → backend)
    print("OK bindings: model load, context, tokenize, decode, logits.")
end

local ok, err = pcall(main)
if not ok then
    io.stderr:write("FAIL: " .. tostring(err) .. "\n")
    os.exit(1)
end
