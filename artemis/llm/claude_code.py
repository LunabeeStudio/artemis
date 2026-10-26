# Copyright 2026 Lunabee Studio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chat model backed by the local Claude Code CLI (``claude -p``).

Lets ARTEMIS run on a Claude subscription instead of API credits: every call
spawns one headless ``claude -p`` process with all built-in tools, MCP servers,
settings and slash commands disabled, so the only thing left is the model.

The CLI has no caller-defined tools, so tool calling is emulated with
structured output: the bound tools are described in the system prompt and the
model answers through a JSON schema ``{"text", "tool_calls": [{name, args}]}``
that ``--json-schema`` enforces. ``with_structured_output`` maps straight onto
``--json-schema``.

Each stream-json input line is a separate CLI turn, so the whole history is
packed into ONE user message with role headers; images stay inline as
Anthropic image blocks.

API-key variables are removed from the child environment: the CLI prefers
``ANTHROPIC_API_KEY`` over the claude.ai login, which would bill API credits.
"""

import asyncio
import json
import os
import shutil
import subprocess
from typing import Any
import uuid
import weakref

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from pydantic import BaseModel

from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

#: Variables that would make the CLI bill the API instead of the subscription.
_API_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")

#: Concurrent ``claude -p`` processes per event loop (env-tunable).
_MAX_CONCURRENCY = int(os.environ.get("ARTEMIS_CLAUDE_CODE_MAX_CONCURRENCY", "4"))
_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)

#: Upper bound for one stdout line (an assistant event can embed large JSON).
_STREAM_LIMIT = 64 * 1024 * 1024

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_TRANSCRIPT_PREAMBLE = (
    "The user message holds the conversation so far as a transcript. Headers mark"
    " each entry: '### user', '### assistant (you)' for your own earlier turns,"
    " '### tool result' for results of your earlier tool calls and '### system note'"
    " for operator instructions. Write the next assistant turn."
)

_TOOL_PROTOCOL = (
    "## How to call tools in this session\n"
    "The tools named in these instructions are NOT native tools here: invoking one"
    " directly fails with 'No such tool available'. Your only native tool is"
    " StructuredOutput. Call it exactly once per turn with your whole answer:"
    " `text` is your message, and `tool_calls` lists every tool you call, in"
    " execution order, as {name, args}. The caller runs those calls back to back"
    " and returns their results in the next turn. Use an empty `tool_calls` list to"
    " call no tool, and never describe a call in `text` instead of listing it.\n\n"
    "Tools you can list in `tool_calls`:\n"
)


class ClaudeCodeError(RuntimeError):
    """A failed ``claude -p`` call; ``status_code`` feeds the reliability classifier."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# --- Serialization -----------------------------------------------------------------------


def _image_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Converts a LangChain image content block into an Anthropic image block."""
    if block.get("type") == "image_url":
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str):
            return None
        if url.startswith("data:") and ";base64," in url:
            header, data = url.split(",", 1)
            media_type = header[len("data:") :].split(";", 1)[0] or "image/jpeg"
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        return {"type": "image", "source": {"type": "url", "url": url}}
    if block.get("type") == "image":
        if isinstance(block.get("source"), dict):
            return {"type": "image", "source": block["source"]}
        data = block.get("data") or block.get("base64")
        if isinstance(data, str):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.get("mime_type") or "image/jpeg",
                    "data": data,
                },
            }
        if isinstance(block.get("url"), str):
            return {"type": "image", "source": {"type": "url", "url": block["url"]}}
    return None


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    """Text and image blocks of a message content; thinking and other blocks are dropped."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for item in content or []:
        if isinstance(item, str):
            if item:
                blocks.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            if item.get("type") == "text" and item.get("text"):
                blocks.append({"type": "text", "text": item["text"]})
            else:
                image = _image_block(item)
                if image is not None:
                    blocks.append(image)
    return blocks


def _header(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "### user"
    if isinstance(message, AIMessage):
        return "### assistant (you)"
    if isinstance(message, ToolMessage):
        name = f" {message.name}" if message.name else ""
        return f"### tool result{name} [{message.tool_call_id}]"
    if isinstance(message, SystemMessage):
        return "### system note"
    return f"### {message.type}"


def split_messages(messages: list[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
    """Splits a history into the system prompt and one user turn's content blocks.

    Leading system messages become the system prompt; everything after them is
    rendered as a transcript in a single user message.
    """
    system_parts: list[str] = []
    index = 0
    while index < len(messages) and isinstance(messages[index], SystemMessage):
        system_parts.extend(
            b["text"] for b in _content_blocks(messages[index].content) if b["type"] == "text"
        )
        index += 1

    blocks: list[dict[str, Any]] = []
    for message in messages[index:]:
        blocks.append({"type": "text", "text": _header(message)})
        blocks.extend(_content_blocks(message.content))
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = [{"name": tc["name"], "args": tc["args"]} for tc in message.tool_calls]
            blocks.append({"type": "text", "text": f"tool_calls: {json.dumps(calls)}"})

    # Adjacent text blocks are merged so headers stay attached to their text.
    merged: list[dict[str, Any]] = []
    for block in blocks:
        if block["type"] == "text" and merged and merged[-1]["type"] == "text":
            merged[-1] = {"type": "text", "text": merged[-1]["text"] + "\n" + block["text"]}
        else:
            merged.append(block)
    return "\n\n".join(system_parts), merged


def _openai_function(tool: Any) -> dict[str, Any]:
    return convert_to_openai_tool(tool)["function"]


def tool_call_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON schema for a turn: free text plus tool calls validated per tool.

    Nested ``$defs`` of every tool are hoisted to the root so their absolute
    ``#/$defs/...`` references still resolve.
    """
    defs: dict[str, Any] = {}
    variants = []
    for tool in tools:
        params = dict(tool.get("parameters") or {"type": "object", "properties": {}})
        for key, value in (params.pop("$defs", None) or {}).items():
            defs.setdefault(key, value)
        params.pop("definitions", None)
        params.pop("title", None)
        variants.append(
            {
                "type": "object",
                "properties": {"name": {"const": tool["name"]}, "args": params},
                "required": ["name", "args"],
            }
        )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "tool_calls": {"type": "array", "items": {"anyOf": variants}},
        },
        "required": ["text", "tool_calls"],
    }
    if defs:
        schema["$defs"] = defs
    return schema


def tool_protocol_prompt(tools: list[dict[str, Any]]) -> str:
    lines = [_TOOL_PROTOCOL]
    for tool in tools:
        params = json.dumps(tool.get("parameters") or {}, ensure_ascii=False)
        lines.append(f"- {tool['name']}: {tool.get('description', '').strip()}\n  params: {params}")
    return "\n".join(lines)


# --- Result parsing ----------------------------------------------------------------------


def parse_events(stdout: str) -> tuple[dict[str, Any] | None, bool]:
    """Returns the CLI ``result`` event and whether a rate limit rejected the call."""
    result = None
    rate_limited = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
        elif event.get("type") == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            rate_limited = rate_limited or info.get("status") == "rejected"
    return result, rate_limited


def raise_for_result(result: dict[str, Any] | None, rate_limited: bool, stderr: str) -> None:
    if result is None:
        tail = stderr.strip()[-500:] or "no result event"
        if "not logged in" in tail.lower() or "/login" in tail:
            raise ClaudeCodeError(
                f"Claude Code is not logged in: run `claude auth login`. ({tail})", 401
            )
        raise ClaudeCodeError(f"claude -p produced no result: {tail}")
    if not result.get("is_error"):
        return
    detail = str(result.get("result") or result.get("subtype") or "unknown error")
    status = result.get("api_error_status")
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    if rate_limited and status is None:
        status = 429
    if "not logged in" in detail.lower() or "/login" in detail:
        raise ClaudeCodeError(
            f"Claude Code is not logged in: run `claude auth login`. ({detail})", 401
        )
    if status == 429 or rate_limited:
        detail = f"rate limit (subscription usage limit): {detail}"
    raise ClaudeCodeError(f"claude -p failed: {detail}", status)


def usage_metadata(result: dict[str, Any]) -> dict[str, Any]:
    usage = result.get("usage") or {}
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    input_tokens = int(usage.get("input_tokens") or 0) + cache_read + cache_creation
    output_tokens = int(usage.get("output_tokens") or 0)
    thinking = int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
        "output_token_details": {"reasoning": thinking},
    }


def _structured(result: dict[str, Any]) -> Any:
    value = result.get("structured_output")
    if value is not None:
        return value
    text = result.get("result")
    if isinstance(text, str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None


# --- Model --------------------------------------------------------------------------------


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max(1, _MAX_CONCURRENCY))
        _SEMAPHORES[loop] = semaphore
    return semaphore


def _child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _API_ENV_VARS}


class ChatClaudeCode(BaseChatModel):
    """LangChain chat model that runs every call through ``claude -p``."""

    model: str = "claude-sonnet-5-5"
    effort: str | None = None
    timeout: float = 120.0
    cli_path: str | None = None
    bound_tools: list[dict[str, Any]] = []
    structured_schema: dict[str, Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort}

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ChatClaudeCode":
        # tool_choice / parallel_tool_calls are not used by ARTEMIS callers.
        return self.model_copy(update={"bound_tools": [_openai_function(t) for t in tools]})

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()

            def parse(message: AIMessage) -> Any:
                return schema.model_validate(message.additional_kwargs["structured_output"])

        else:
            json_schema = (
                schema
                if isinstance(schema, dict) and "properties" in schema
                else convert_to_json_schema(schema)
            )

            def parse(message: AIMessage) -> Any:
                return message.additional_kwargs["structured_output"]

        bound = self.model_copy(update={"structured_schema": json_schema, "bound_tools": []})
        return bound | RunnableLambda(parse)

    # -- request ---------------------------------------------------------------------------

    def _command(self, system_prompt: str, schema: dict[str, Any] | None) -> list[str]:
        cli = self.cli_path or shutil.which("claude")
        if not cli:
            raise ClaudeCodeError(
                "Claude Code CLI not found on PATH: install it and run `claude auth login`.", 401
            )
        command = [
            cli,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--model",
            self.model,
            "--system-prompt",
            system_prompt,
        ]
        if self.effort:
            command += ["--effort", self.effort]
        if schema is not None:
            command += ["--json-schema", json.dumps(schema)]
        return command

    def _request(self, messages: list[BaseMessage]) -> tuple[list[str], bytes]:
        system, blocks = split_messages(messages)
        parts = [system or _DEFAULT_SYSTEM_PROMPT, _TRANSCRIPT_PREAMBLE]
        schema = self.structured_schema
        if self.bound_tools and schema is None:
            parts.append(tool_protocol_prompt(self.bound_tools))
            schema = tool_call_schema(self.bound_tools)
        if not blocks:
            blocks = [{"type": "text", "text": "### user\n(continue)"}]
        line = {"type": "user", "message": {"role": "user", "content": blocks}}
        payload = (json.dumps(line) + "\n").encode("utf-8")
        return self._command("\n\n".join(parts), schema), payload

    def _message(self, result: dict[str, Any]) -> AIMessage:
        usage = usage_metadata(result)
        metadata = {
            "model_name": self.model,
            "total_cost_usd": result.get("total_cost_usd"),
            "stop_reason": result.get("stop_reason"),
            "duration_ms": result.get("duration_ms"),
            "num_turns": result.get("num_turns"),
        }
        # A clean structured answer takes 2 CLI turns (StructuredOutput + its
        # result); more means the model tried native tools or failed validation.
        if (self.bound_tools or self.structured_schema) and (result.get("num_turns") or 0) > 2:
            logger.warning(
                f"claude -p ({self.model}) needed {result['num_turns']} turns for one answer"
            )
        if self.structured_schema is not None:
            structured = _structured(result)
            if structured is None:
                raise ClaudeCodeError("claude -p returned no structured output")
            return AIMessage(
                content=json.dumps(structured, ensure_ascii=False),
                additional_kwargs={"structured_output": structured},
                usage_metadata=usage,
                response_metadata=metadata,
            )
        if self.bound_tools:
            structured = _structured(result) or {}
            tool_calls = [
                {
                    "name": call["name"],
                    "args": call.get("args") or {},
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "tool_call",
                }
                for call in structured.get("tool_calls") or []
                if isinstance(call, dict) and call.get("name")
            ]
            return AIMessage(
                content=str(structured.get("text") or ""),
                tool_calls=tool_calls,
                usage_metadata=usage,
                response_metadata=metadata,
            )
        return AIMessage(
            content=str(result.get("result") or ""),
            usage_metadata=usage,
            response_metadata=metadata,
        )

    # -- execution -------------------------------------------------------------------------

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        command, payload = self._request(messages)
        async with _semaphore():
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_child_env(),
                limit=_STREAM_LIMIT,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(payload), timeout=self.timeout
                )
            except (TimeoutError, asyncio.CancelledError):
                process.kill()
                await process.wait()
                raise
        result, rate_limited = parse_events(stdout.decode("utf-8", "replace"))
        raise_for_result(result, rate_limited, stderr.decode("utf-8", "replace"))
        return ChatResult(generations=[ChatGeneration(message=self._message(result))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        command, payload = self._request(messages)
        completed = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            env=_child_env(),
            timeout=self.timeout,
            check=False,
        )
        result, rate_limited = parse_events(completed.stdout.decode("utf-8", "replace"))
        raise_for_result(result, rate_limited, completed.stderr.decode("utf-8", "replace"))
        return ChatResult(generations=[ChatGeneration(message=self._message(result))])


# --- Diagnostics --------------------------------------------------------------------------


def login_status(timeout: float = 15.0) -> tuple[bool, str]:
    """Whether the local CLI is logged in with a claude.ai (subscription) account.

    Returns ``(ok, detail)``; ``detail`` names the account or the problem.
    """
    cli = shutil.which("claude")
    if not cli:
        return False, "Claude Code CLI not found on PATH"
    try:
        completed = subprocess.run(
            [cli, "auth", "status"],
            capture_output=True,
            env=_child_env(),
            timeout=timeout,
            check=False,
        )
        status = json.loads(completed.stdout.decode("utf-8", "replace") or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return False, f"`claude auth status` failed: {exc}"
    if not status.get("loggedIn"):
        return False, "Claude Code is not logged in: run `claude auth login`"
    if status.get("authMethod") != "claude.ai":
        return False, f"Claude Code uses {status.get('authMethod')!r}, not a claude.ai subscription"
    return (
        True,
        f"claude.ai login ({status.get('orgName') or status.get('email') or 'subscription'})",
    )


def uses_claude_code(llm_config: Any) -> bool:
    """True when any node of an LLMConfig (or its fallback) runs on ``claude_code``."""
    try:
        data = llm_config.model_dump()
    except AttributeError:
        return False

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            if str(value.get("provider") or "").lower().replace("-", "_") == "claude_code":
                return True
            return any(walk(v) for v in value.values())
        return False

    return walk(data)
