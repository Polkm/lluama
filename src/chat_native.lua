-- Native chat template API (llama_chat_apply_template, llama_chat_builtin_templates)

local ffi = require("ffi")

return function(llama)
	-- Apply template string to messages. messages = { { role = "user"|"assistant"|"system", content = "..." }, ... }
	-- add_ass: add assistant turn at the end. Returns formatted string or nil on error.
	local function chat_apply_template(tmpl, messages, add_ass)
		local n = #messages
		if n == 0 then
			return llama.llama_chat_apply_template(tmpl, nil, 0, add_ass and true or false, nil, 0) == 0 and "" or nil
		end
		local msg_arr = ffi.new("llama_chat_message[?]", n)
		for i = 0, n - 1 do
			local m = messages[i + 1]
			msg_arr[i].role = m.role or "user"
			msg_arr[i].content = m.content or ""
		end
		local cap = 65536
		local buf = ffi.new("char[?]", cap)
		local len = llama.llama_chat_apply_template(tmpl, msg_arr, n, add_ass and true or false, buf, cap)
		if len < 0 then return nil end
		if len > cap then
			buf = ffi.new("char[?]", len)
			len = llama.llama_chat_apply_template(tmpl, msg_arr, n, add_ass and true or false, buf, len)
			if len < 0 then return nil end
		end
		-- Return value is bytes written including null terminator (C string convention).
		return ffi.string(buf, len > 0 and (len - 1) or 0)
	end

	-- Return the length in bytes that apply_template would write (for prev_len slice logic).
	-- Pass buf=nil, length=0; returns required size or negative on error.
	function chat_apply_template_length(tmpl, messages, add_ass)
		local n = #messages
		if n == 0 then
			return llama.llama_chat_apply_template(tmpl, nil, 0, add_ass and true or false, nil, 0)
		end
		local msg_arr = ffi.new("llama_chat_message[?]", n)
		for i = 0, n - 1 do
			local m = messages[i + 1]
			msg_arr[i].role = m.role or "user"
			msg_arr[i].content = m.content or ""
		end
		return llama.llama_chat_apply_template(tmpl, msg_arr, n, add_ass and true or false, nil, 0)
	end

	-- List built-in template names. Returns { "name1", "name2", ... }.
	local function chat_builtin_templates()
		local buf = ffi.new("const char*[?]", 64)
		local n = llama.llama_chat_builtin_templates(buf, 64)
		if n <= 0 then return {} end
		local out = {}
		for i = 0, n - 1 do
			if buf[i] ~= nil then
				out[i + 1] = ffi.string(buf[i])
			end
		end
		return out
	end

	return {
		chat_apply_template = chat_apply_template,
		chat_apply_template_length = chat_apply_template_length,
		chat_builtin_templates = chat_builtin_templates,
	}
end
