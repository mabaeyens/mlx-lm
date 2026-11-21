# Copyright © 2026 Apple Inc.
"""Inkling 2 tool calls.

The model writes each call as a message whose header carries the function name
and whose body is JSON:

    <|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Madrid"}}<|end_message|>

The server hands this parser the text buffered while in the tool state, which
starts at the content marker (the header before it is not part of it) and can
hold several calls back to back.
"""

import json
import re

tool_call_start = "<|content_invoke_tool_json|>"

tool_call_end = "<|end_message|>"

_CALL = re.compile(
    re.escape(tool_call_start) + r"(.*?)(?:" + re.escape(tool_call_end) + r"|$)",
    re.DOTALL,
)


def _one(text):
    call = json.loads(text.strip())
    return {"name": call["name"], "arguments": call.get("args") or {}}


def parse_tool_call(text, tools=None):
    bodies = [b for b in _CALL.findall(text) if b.strip()]
    if not bodies:
        # No marker: the whole buffer is the JSON body.
        return _one(text.replace(tool_call_end, ""))
    calls = [_one(b) for b in bodies]
    return calls[0] if len(calls) == 1 else calls
