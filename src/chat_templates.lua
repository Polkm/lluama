-- Chat Templates for different LLM models
-- Each template defines the format for system, user, and assistant messages

local templates = {}

-- Phi-3 (Microsoft)
-- Format: <|system|>\nSystem<|end|>\n<|user|>\nUser<|end|>\n<|assistant|>Response<|end|>
templates.phi3 = {
	name = "phi3",
	system_start = "<|system|>\n",
	system_end = "<|end|>\n",
	user_start = "<|user|>\n",
	user_end = "<|end|>\n",
	assistant_start = "<|assistant|>",
	assistant_end = "<|end|>\n",
	-- Stop sequences that indicate end of response
	stop_sequences = { "<|end|>", "<|user|>", "<|assistant|>" },
}

-- Llama 2 (Meta)
-- Format: [INST] <<SYS>>\nSystem\n<</SYS>>\n\nUser [/INST] Response
templates.llama2 = {
	name = "llama2",
	system_start = "[INST] <<SYS>>\n",
	system_end = "\n<</SYS>>\n\n",
	user_start = "",  -- System already includes [INST], subsequent turns use [INST]
	user_end = " [/INST] ",
	assistant_start = "",
	assistant_end = " </s><s>[INST] ",
	stop_sequences = { "</s>", "[INST]" },
	-- Special: first user message is part of system block
	first_user_in_system = true,
}

-- Llama 3 / Llama 3.1 (Meta)
-- Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSystem<|eot_id|>
templates.llama3 = {
	name = "llama3",
	bos = "<|begin_of_text|>",
	system_start = "<|start_header_id|>system<|end_header_id|>\n\n",
	system_end = "<|eot_id|>",
	user_start = "<|start_header_id|>user<|end_header_id|>\n\n",
	user_end = "<|eot_id|>",
	assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n\n",
	assistant_end = "<|eot_id|>",
	stop_sequences = { "<|eot_id|>", "<|end_of_text|>" },
}

-- Gemma (Google)
-- Format: <start_of_turn>user\nUser<end_of_turn>\n<start_of_turn>model\nResponse<end_of_turn>
templates.gemma = {
	name = "gemma",
	system_start = "<start_of_turn>user\n",  -- Gemma uses user turn for system
	system_end = "<end_of_turn>\n",
	user_start = "<start_of_turn>user\n",
	user_end = "<end_of_turn>\n",
	assistant_start = "<start_of_turn>model\n",
	assistant_end = "<end_of_turn>\n",
	stop_sequences = { "<end_of_turn>", "<start_of_turn>" },
}

-- Mistral / Mixtral
-- Format: [INST] User [/INST] Response</s> [INST] User [/INST]
templates.mistral = {
	name = "mistral",
	system_start = "[INST] ",  -- System goes at start of first instruction
	system_end = "\n\n",
	user_start = "",
	user_end = " [/INST] ",
	assistant_start = "",
	assistant_end = "</s> [INST] ",
	stop_sequences = { "</s>", "[INST]" },
}

-- ChatML (OpenAI style, used by many fine-tunes)
-- Format: <|im_start|>system\nSystem<|im_end|>\n<|im_start|>user\nUser<|im_end|>\n<|im_start|>assistant\n
templates.chatml = {
	name = "chatml",
	system_start = "<|im_start|>system\n",
	system_end = "<|im_end|>\n",
	user_start = "<|im_start|>user\n",
	user_end = "<|im_end|>\n",
	assistant_start = "<|im_start|>assistant\n",
	assistant_end = "<|im_end|>\n",
	stop_sequences = { "<|im_end|>", "<|im_start|>" },
}

-- Qwen (Alibaba)
templates.qwen = {
	name = "qwen",
	system_start = "<|im_start|>system\n",
	system_end = "<|im_end|>\n",
	user_start = "<|im_start|>user\n",
	user_end = "<|im_end|>\n",
	assistant_start = "<|im_start|>assistant\n",
	assistant_end = "<|im_end|>\n",
	-- Include "<|im_end|" so we stop when tokenized as two tokens ("<|im_end|" then ">")
	stop_sequences = { "<|im_end|>", "<|im_end|", "<|im_start|>", "<|endoftext|>" },
}

-- Alpaca style (simple instruction format)
templates.alpaca = {
	name = "alpaca",
	system_start = "### System:\n",
	system_end = "\n\n",
	user_start = "### Instruction:\n",
	user_end = "\n\n",
	assistant_start = "### Response:\n",
	assistant_end = "\n\n",
	stop_sequences = { "### Instruction:", "### System:" },
}

-- Format a conversation using a template
-- history: array of { role = "user"|"assistant", content = "..." }
-- system_prompt: optional system prompt string
-- template: one of the templates above
function templates.format_conversation(history, system_prompt, template)
	local formatted = ""
	
	-- Add BOS token if template has one
	if template.bos then
		formatted = template.bos
	end
	
	-- Add system prompt if provided
	if system_prompt and #system_prompt > 0 then
		formatted = formatted .. template.system_start .. system_prompt .. template.system_end
	end
	
	-- Add conversation history
	for _, msg in ipairs(history) do
		if msg.role == "user" then
			formatted = formatted .. template.user_start .. msg.content .. template.user_end .. template.assistant_start
		elseif msg.role == "assistant" then
			formatted = formatted .. msg.content .. template.assistant_end
		end
	end
	
	-- If last message was assistant, add assistant start for next response
	if #history > 0 and history[#history].role == "assistant" then
		formatted = formatted .. template.assistant_start
	end
	
	return formatted
end

-- Clean response by removing stop sequences and trimming
function templates.clean_response(text, template)
	for _, seq in ipairs(template.stop_sequences) do
		local pos = text:find(seq, 1, true)  -- plain text search
		if pos then
			text = text:sub(1, pos - 1)
		end
	end
	-- Trim whitespace
	text = text:match("^%s*(.-)%s*$") or text
	return text
end

-- Check if text contains a stop sequence, return (should_stop, cleaned_text)
function templates.check_stop(text, template)
	for _, seq in ipairs(template.stop_sequences) do
		local pos = text:find(seq, 1, true)
		if pos then
			return true, text:sub(1, pos - 1)
		end
	end
	return false, text
end

-- Remove from the end of text any suffix that is a prefix of a stop sequence (e.g. trailing "<|" before "<|im_end|>").
function templates.trim_trailing_stop_prefix(text, template)
	local out = text
	local changed = true
	while changed do
		changed = false
		for _, seq in ipairs(template.stop_sequences) do
			for k = #seq - 1, 1, -1 do
				if #out >= k and out:sub(-k) == seq:sub(1, k) then
					out = out:sub(1, #out - k)
					changed = true
					break
				end
			end
			if changed then break end
		end
	end
	return out
end

-- Get template by name (case-insensitive)
function templates.get(name)
	name = name:lower()
	return templates[name]
end

-- List available templates
function templates.list()
	local names = {}
	for k, v in pairs(templates) do
		if type(v) == "table" and v.name then
			table.insert(names, v.name)
		end
	end
	table.sort(names)
	return names
end

return templates
