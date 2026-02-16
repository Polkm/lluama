-- Minimal unit test runner. Run from repo root: luajit test/unit/run.lua
-- Runs all *_spec.lua files in test/unit/ and reports pass/fail.

package.path = "./?.lua;" .. package.path

local unit_dir = "test/unit"
if arg and arg[0] then
	local p = arg[0]:gsub("\\", "/")
	local dir = p:match("^(.*)/") or "."
	unit_dir = dir
end

local function run_file(path)
	local name = path:match("([^/\\]+)$") or path
	local ok, err = pcall(dofile, path)
	if ok then
		return true, name
	end
	return false, name, tostring(err)
end

local spec_files = { "chat_templates_spec.lua", "lluama_spec.lua", "sampler_spec.lua", "chat_session_spec.lua" }
local passed, failed = 0, 0
local failures = {}

for _, name in ipairs(spec_files) do
	local path = unit_dir .. "/" .. name
	local f = io.open(path, "r")
	if f then
		f:close()
		local ok, file, err = run_file(path)
		if ok then
			passed = passed + 1
			print("[PASS] " .. file)
		else
			failed = failed + 1
			table.insert(failures, { file = file, err = err })
			print("[FAIL] " .. file .. ": " .. (err or ""))
		end
	else
		-- skip missing
	end
end

print("")
print(string.format("%d passed, %d failed", passed, failed))
if failed > 0 then
	for _, f in ipairs(failures) do
		io.stderr:write(f.file .. ": " .. f.err .. "\n")
	end
	os.exit(1)
end
