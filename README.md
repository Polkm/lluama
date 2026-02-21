# lluama

Lua (LuaJIT) FFI bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp). Load GGUF models, run inference, and use chat templates from Lua.

## Requirements

- **LuaJIT** (with FFI) — e.g. [LuaJIT](https://luajit.org/) 2.0 or 2.1
- **llama.cpp** — shared library and GGML backends on your system. Install and put `llama`/`ggml` on your library path (or PATH on Windows) however you prefer.
- **A GGUF model** — for the test script, place one at the path used in `test/test_bindings.lua` or edit that path.

**Compatibility:** The bindings are maintained against the [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) API (batch, sampler chain, chat templates, context params). Single-file model loading uses `llama_model_load_from_file` when available and falls back to deprecated `llama_load_model_from_file` for older builds. When upgrading llama.cpp, check release notes for renames or signature changes and update `src/headers/llama_ffi.h` if needed.

## Running the test

From the repo root, with llama.cpp and a GGUF model available:

```bash
luajit test/test_bindings.lua
```

The test loads the model path defined in the script, runs backend init, tokenizes "Hello", and performs one decode step.

**CLI chat:** `luajit examples/chat.lua [model_path]` — minimal back-and-forth chat (default model path as in test). Empty line to quit.

**CLI chat (JSON):** `luajit examples/chat_json.lua [model_path]` — same as above with `grammar = "json"` so output is constrained to valid JSON.

**Demo chat:** `luajit examples/chat_demo.lua [--show-probs] [--context-length N] [model_path]` — progress bar on load, optional top-5 token probs after each reply, `/embed <text>` for one-shot embedding, `/probs` to toggle probs. Uses `progress_callback` and optional `kv_overrides` (e.g. context length).

**Unit tests** (no model or DLL required for most): `luajit test/unit/run.lua` — runs lluama, Sampler, and ChatSession specs.

**Integration tests** (Qwen model required): `luajit test/integration_test.lua [model_path] [--loop N]` — load model, tokenize, decode, sampler loop, two turns. Use `--loop 10` to run repeatedly and catch flakes. Call `ctx:set_sampler(sampler)` before `ctx:decode_tokens(...)` so the backend associates logits with the sequence.

## Project layout

- `src/lluama.lua` — main module (require this)
- `src/bindings.lua` — FFI bindings (loads `llama` and `ggml` from headers in `src/headers/`)
- `src/backend.lua` — Backend class (init, __gc)
- `src/model.lua` — Model class (context, tokenize, __gc)
- `src/context.lua` — Context class (decode_tokens, decode_one, logits, set_sampler, __gc)
- `src/sampler.lua` — Sampler class (temp/dist/top_p chain, accept, sample, __gc)
- `src/chat_session.lua` — ChatSession (prompt + generate with template; stop strings from model vocab)
- `examples/chat.lua`, `examples/chat_demo.lua`, `examples/chat_grammar.lua`, `examples/chat_json.lua` — CLI chat examples (run from repo root)
- `test/test_bindings.lua` — minimal test: load model, tokenize, one decode step
- `test/unit/run.lua` — unit test runner; `test/unit/helper.lua` — shared scaffolding; `test/unit/*_spec.lua` — specs (no model required)

## Usage

Require the library; all `llama_*` and `ggml_*` functions live on the returned table.

**Classes:** `Backend`, `Model`, `Context`, `Sampler`, `ChatSession`. Ownership: context holds model, model holds backend; attach a `Sampler` to a context for sampling. Use a given context from a single thread unless the llama.cpp library documents otherwise.

**Simple chat (recommended):** use `ChatSession` so the library handles templates, decode positions, and stop logic.

```lua
local lluama = require("src.lluama")
local backend = lluama.Backend()
backend:init()

local session = lluama.ChatSession(backend, "path/to/model.gguf", {
  -- template = "auto",        -- optional: auto-detect from model (default); or "chatml", "qwen", "llama3", etc.
  system_prompt = "You are a helpful assistant.",
  temp = 0.7,
  grammar = "json",            -- optional: force output to be valid JSON
})
session:prompt("Hello!")
session:generate(256, function(piece) io.write(piece) io.flush() end)  -- stream
-- Or: local reply = session:generate(256)  -- no callback = return full reply
```

**JSON output:** Pass `grammar = "json"` in `ChatSession` opts or when creating a `Sampler` (then pass the model as second argument: `lluama.Sampler({ temp = 0.7, grammar = "json" }, model)`). Custom GBNF strings are also supported via `grammar = "<gbnf string>"` and optional `grammar_root = "root"`.

**Lower-level:** `Model`, `Context`, `Sampler` for custom loops.

```lua
local model = lluama.Model(backend, "path/to/model.gguf")
local ctx = model:context({ n_ctx = 512, n_batch = 256 })
local sampler = lluama.Sampler({ temp = 0.7, seed = 12345 })
-- Or with JSON grammar: lluama.Sampler({ temp = 0.7, grammar = "json" }, model)
ctx:set_sampler(sampler)
local token_ids = model:tokenize("Hello", false)
local err = ctx:decode_tokens(token_ids, 0)  -- second arg: pos_start (for multi-turn)
-- ctx:decode_one(token, pos), sampler:accept(token), sampler:sample(ctx.ctx, logits_idx)
-- model:token_to_piece(token) — decode a token id to its string piece
-- ctx:sampled_top_k_ith(0, 5) — after sampling, get top-5 token ids and probs for the last output
```

**Embeddings:** For embedding models, create a context with `embeddings = true` and optional `pooling_type` (`"mean"`, `"cls"`, `"last"`, `"rank"`), then `encode` a batch and read `ctx:embeddings_seq(seq_id)`. One-shot helper: `model:embed("text to embed", { pooling_type = "mean" })` returns a Lua array of floats (pooled embedding vector).

**Classifier / reranker:** Use a context with `embeddings = true` and `pooling_type = "rank"`. After encoding, `ctx:embeddings_seq(seq_id)` returns the rank scores. For classifier models, `model:n_cls_out()` and `model:cls_label(i)` give the number of classes and label strings.

**Model load options:** `lluama.Model(backend, path, { progress_callback = function(p) print(p) end })` reports load progress (0.0–1.0); return `true` from the callback to abort. `kv_overrides = { { key = "context_length", type = "int", value = 8192 } }` overrides model metadata at load (types: `"int"`, `"float"`, `"bool"`, `"str"`).

**NUMA:** On multi-socket systems, call `backend:numa_init(strategy)` before `backend:init()` (e.g. strategy `0` = disabled, see ggml NUMA docs).

**LoRA:** `lluama.AdapterLora(model, path_lora)` then `ctx:set_adapter_lora(adapter[, scale])` / `ctx:rm_adapter_lora(adapter)` / `ctx:clear_adapter_lora()`. Adapters have no explicit free in the current API; keep them alive while in use.

Raw cdata: use `model.model` and `ctx.ctx` for direct `lluama.llama.llama_*` / `lluama.ggml.ggml_*` calls.

**Logging:** Logs are silent by default. To hook in: `lluama.set_log_callback(function(level, text) ... end)`. Pass `nil` to keep silent.

## License

MIT — see [LICENSE](LICENSE).
