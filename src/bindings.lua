-- FFI bindings for llama.cpp and GGML
-- Loads FFI definitions from .h files at runtime

local ffi = require("ffi")

local function load_ffi_lib(lib_name, header_name)
    local lib = ffi.load(lib_name)
    -- Resolve headers dir: same directory as this file, under "headers/"
    local source = (debug.getinfo(1, "S") or {}).source
    if type(source) == "string" and source:sub(1, 1) == "@" then
        source = source:sub(2):gsub("\\", "/")
    else
        source = ""
    end
    local base = source:match("^(.+)/[^/]+$") or "."
    local header_path = base .. "/headers/" .. header_name
    local header_file = io.open(header_path, "r")
    if not header_file then
        -- Fallback: src/headers/ from current working directory
        header_path = "src/headers/" .. header_name
        header_file = io.open(header_path, "r")
    end
    if not header_file then
        error("Failed to open " .. header_name .. ". Tried: " .. base .. "/headers/" .. header_name .. " and src/headers/" .. header_name)
    end
    local content = header_file:read("*all")
    header_file:close()
    ffi.cdef(content)
    return lib
end

return {
    llama = load_ffi_lib("llama", "llama_ffi.h"),
    ggml = load_ffi_lib("ggml", "ggml_ffi.h"),
}
