# lluama

Lua (LuaJIT) FFI bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp). Load GGUF models, run inference, and use chat templates from Lua.

## Requirements

- **LuaJIT** (with FFI) — e.g. [LuaJIT](https://luajit.org/) 2.0 or 2.1
- **llama.cpp** — shared library and GGML backends on your system. Install and put `llama`/`ggml` on your library path (or PATH on Windows) however you prefer.
- **A GGUF model** — for the test script, place one at the path used in `test/test_bindings.lua` or edit that path.

## Running the test

From the repo root, with llama.cpp and a GGUF model available:

```bash
luajit test/test_bindings.lua
```

The test loads the model path defined in the script, runs backend init, tokenizes "Hello", and performs one decode step.

**CLI chat:** `luajit test/chat.lua [model_path]` — minimal back-and-forth chat (default model path as in test). Empty line to quit.

**Unit tests** (no model or DLL required for most): `luajit test/unit/run.lua` — runs chat_templates, lluama, and Sampler specs.

**Integration tests** (Qwen model required): `luajit test/integration_test.lua [model_path] [--loop N]` — load model, tokenize, decode, sampler loop, two turns. Use `--loop 10` to run repeatedly and catch flakes. Call `ctx:set_sampler(sampler)` before `ctx:decode_tokens(...)` so the backend associates logits with the sequence.

## Project layout

- `src/lluama.lua` — main module (require this)
- `src/bindings.lua` — FFI bindings (loads `llama` and `ggml` from headers in `src/headers/`)
- `src/backend.lua` — Backend class (init, __gc)
- `src/model.lua` — Model class (context, tokenize, __gc)
- `src/context.lua` — Context class (decode_tokens, decode_one, logits, set_sampler, __gc)
- `src/sampler.lua` — Sampler class (temp/dist/top_p chain, accept, sample, __gc)
- `src/chat_templates.lua` — chat templates (Phi-3, Llama 2/3, Qwen, etc.)
- `test/test_bindings.lua` — minimal test: load model, tokenize, one decode step
- `test/unit/run.lua` — unit test runner; `test/unit/*_spec.lua` — specs (no model required)

## Usage

Require the library; all `llama_*` and `ggml_*` functions live on the returned table.

**Classes:** `Backend`, `Model`, `Context`, `Sampler` with `__gc` so cleanup is automatic. Ownership: context holds model, model holds backend; attach a `Sampler` to a context for sampling.

```lua
local lluama = require("src.lluama")

local backend = lluama.Backend()
backend:init()

local model = lluama.Model(backend, "path/to/model.gguf")
local ctx = model:context({ n_ctx = 512, n_batch = 256 })

-- Optional: sampler for generation (temp + dist, optional top_p). Set before decode.
local sampler = lluama.Sampler({ temp = 0.7, seed = 12345 })
ctx:set_sampler(sampler)

local token_ids = model:tokenize("Hello", false)
local err = ctx:decode_tokens(token_ids)  -- optional second arg: pos_start (for multi-turn)
-- Single-token decode (e.g. in a loop): ctx:decode_one(token, pos)
local logits = ctx:logits()

-- Model accessors: model:vocab(), model:n_params(), model:desc()
-- Sampler: sampler:accept(token), sampler:sample(ctx.ctx, logits_idx), sampler:reset()
```

Raw cdata: use `model.model` and `ctx.ctx` for direct `lluama.llama.llama_*` / `lluama.ggml.ggml_*` calls.

**Logging:** Logs are silent by default. To hook in: `lluama.set_log_callback(function(level, text) ... end)`. Pass `nil` to keep silent.

## License

MIT — see [LICENSE](LICENSE).
