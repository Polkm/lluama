-- Unit tests for src.chat_templates (no model required)

package.path = "./?.lua;" .. package.path
local t = require("src.chat_templates")

local function assert_eq(a, b, msg)
	msg = msg or ("expected " .. tostring(a) .. " == " .. tostring(b))
	assert(a == b, msg)
end

local function assert_str_contains(s, sub, msg)
	assert(s and s:find(sub, 1, true), msg or ("expected string to contain " .. tostring(sub)))
end

-- get / list
do
	local names = t.list()
	assert(type(names) == "table", "list() returns table")
	assert(#names >= 1, "at least one template")
	assert(t.get("qwen"), "get qwen")
	assert(t.get("Qwen"), "get Qwen (case-insensitive)")
	assert(t.get("QWEN").name == "qwen", "get returns template with .name")
	assert(t.get("nonexistent") == nil, "get nonexistent returns nil")
end

-- format_conversation
do
	local q = t.get("qwen")
	local hist = { { role = "user", content = "Hi" } }
	local out = t.format_conversation(hist, nil, q)
	assert_str_contains(out, "<|im_start|>user\n")
	assert_str_contains(out, "Hi")
	assert_str_contains(out, "<|im_end|>\n")
	assert_str_contains(out, "<|im_start|>assistant\n")

	local with_sys = t.format_conversation(hist, "You are helpful.", q)
	assert_str_contains(with_sys, "<|im_start|>system\n")
	assert_str_contains(with_sys, "You are helpful.")

	local multi = t.format_conversation({
		{ role = "user", content = "One" },
		{ role = "assistant", content = "Two" },
		{ role = "user", content = "Three" },
	}, nil, q)
	assert_str_contains(multi, "One")
	assert_str_contains(multi, "Two")
	assert_str_contains(multi, "Three")
end

-- check_stop
do
	local q = t.get("qwen")
	local stop, cleaned = t.check_stop("hello world", q)
	assert(stop == false)
	assert(cleaned == "hello world")

	stop, cleaned = t.check_stop("hello <|im_end|> rest", q)
	assert(stop == true)
	assert(cleaned == "hello ")
end

-- clean_response
do
	local q = t.get("qwen")
	local s = "  answer <|im_end|> trailing  "
	local c = t.clean_response(s, q)
	assert_str_contains(c, "answer")
	assert(not c:find("<|im_end|>", 1, true), "stop sequence removed")
end

print("chat_templates_spec: all passed")
