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

"""Unit tests for the Claude Code CLI chat model (provider ``claude_code``)."""

import json
from pathlib import Path
import stat

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
import pytest

from artemis.core.tool_declaration import ToolDeclaration
from artemis.llm.claude_code import (
    ChatClaudeCode,
    ClaudeCodeError,
    parse_events,
    raise_for_result,
    split_messages,
    tool_call_schema,
    usage_metadata,
    uses_claude_code,
)
from artemis.llm.reliability import FailureCategory, classify_failure
from artemis.llm.router import ModelEndpoint, ModelFactory, ModelProvider
from artemis.tools.tool_wrapper import resolve_image_carrier

PNG_URL = "data:image/png;base64,iVBORw0KGgo="

RESULT_EVENT = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "",
    "stop_reason": "end_turn",
    "total_cost_usd": 0.01,
    "usage": {
        "input_tokens": 2,
        "cache_creation_input_tokens": 300,
        "cache_read_input_tokens": 1000,
        "output_tokens": 50,
        "output_tokens_details": {"thinking_tokens": 20},
    },
}


def _tap_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="tap",
        description="Tap a point.",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    )


def _stdout(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_split_messages_packs_history_into_one_turn():
    messages = [
        SystemMessage(content="SYS"),
        HumanMessage(
            content=[
                {"type": "text", "text": "goal"},
                {"type": "image_url", "image_url": {"url": PNG_URL}},
            ]
        ),
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "tapping"},
            ],
            tool_calls=[{"name": "tap", "args": {"x": 1, "y": 2}, "id": "c1"}],
        ),
        ToolMessage(content="ok", tool_call_id="c1", name="tap"),
        SystemMessage(content="last turn"),
    ]
    system, blocks = split_messages(messages)

    assert system == "SYS"
    assert [b["type"] for b in blocks] == ["text", "image", "text"]
    assert blocks[0]["text"] == "### user\ngoal"
    assert blocks[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgo=",
    }
    tail = blocks[2]["text"]
    assert "### assistant (you)\ntapping" in tail
    assert 'tool_calls: [{"name": "tap", "args": {"x": 1, "y": 2}}]' in tail
    assert "### tool result tap [c1]\nok" in tail
    assert "### system note\nlast turn" in tail
    assert "hidden" not in tail


def test_bind_tools_accepts_declarations_and_langchain_tools():
    class Swipe(BaseModel):
        direction: str

    swipe = StructuredTool.from_function(
        func=lambda direction: direction, name="swipe", description="Swipe.", args_schema=Swipe
    )
    model = ChatClaudeCode().bind_tools([_tap_declaration(), swipe])

    assert [t["name"] for t in model.bound_tools] == ["tap", "swipe"]
    schema = tool_call_schema(model.bound_tools)
    variants = schema["properties"]["tool_calls"]["items"]["anyOf"]
    assert variants[0]["properties"]["name"] == {"const": "tap"}
    assert variants[1]["properties"]["args"]["required"] == ["direction"]


def test_tool_call_schema_hoists_nested_defs():
    tool = {
        "name": "act",
        "parameters": {
            "type": "object",
            "properties": {"target": {"$ref": "#/$defs/Target"}},
            "$defs": {"Target": {"type": "object", "properties": {"id": {"type": "string"}}}},
        },
    }
    schema = tool_call_schema([tool])
    assert "Target" in schema["$defs"]
    args = schema["properties"]["tool_calls"]["items"]["anyOf"][0]["properties"]["args"]
    assert "$defs" not in args


def test_request_command_disables_tools_and_keeps_system_prompt():
    model = ChatClaudeCode(model="claude-haiku-5-5", effort="low", cli_path="/bin/claude")
    command, payload = model.bind_tools([_tap_declaration()])._request(
        [SystemMessage(content="SYS"), HumanMessage(content="go")]
    )

    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--model") + 1] == "claude-haiku-5-5"
    assert command[command.index("--effort") + 1] == "low"
    system_prompt = command[command.index("--system-prompt") + 1]
    assert system_prompt.startswith("SYS") and "- tap: Tap a point." in system_prompt
    assert json.loads(command[command.index("--json-schema") + 1])["required"] == [
        "text",
        "tool_calls",
    ]
    line = json.loads(payload)
    assert line["type"] == "user" and line["message"]["content"][0]["text"] == "### user\ngo"


def test_message_maps_structured_output_to_tool_calls_and_usage():
    model = ChatClaudeCode().bind_tools([_tap_declaration()])
    event = {
        **RESULT_EVENT,
        "structured_output": {
            "text": "tap it",
            "tool_calls": [{"name": "tap", "args": {"x": 3, "y": 4}}],
        },
    }
    message = model._message(event)

    assert message.content == "tap it"
    assert [(c["name"], c["args"]) for c in message.tool_calls] == [("tap", {"x": 3, "y": 4})]
    assert message.usage_metadata["input_tokens"] == 1302
    assert message.usage_metadata["input_token_details"]["cache_read"] == 1000
    assert message.usage_metadata["output_token_details"]["reasoning"] == 20
    assert usage_metadata(event)["total_tokens"] == 1352


def test_parse_events_and_errors():
    result, limited = parse_events(_stdout({"type": "system"}, RESULT_EVENT))
    assert result["subtype"] == "success" and not limited
    raise_for_result(result, limited, "")

    rejected = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}}
    failed = {**RESULT_EVENT, "is_error": True, "result": "usage limit reached"}
    result, limited = parse_events(_stdout(rejected, failed))
    with pytest.raises(ClaudeCodeError) as excinfo:
        raise_for_result(result, limited, "")
    assert excinfo.value.status_code == 429
    assert classify_failure(excinfo.value).category == FailureCategory.RATE_LIMIT

    with pytest.raises(ClaudeCodeError) as excinfo:
        raise_for_result(None, False, "Not logged in · Please run /login")
    assert excinfo.value.status_code == 401


def test_provider_wiring():
    assert ModelProvider.from_string("claude_code") == ModelProvider.CLAUDE_CODE
    assert ModelProvider.from_string("claude-code") == ModelProvider.CLAUDE_CODE
    model = ModelFactory.create_model(
        ModelEndpoint(
            provider=ModelProvider.CLAUDE_CODE,
            model_name="claude-opus-5-5",
            thinking_level="minimal",
            timeout_seconds=90,
        )
    )
    assert isinstance(model, ChatClaudeCode)
    assert (model.model, model.effort, model.timeout) == ("claude-opus-5-5", "low", 90)
    assert resolve_image_carrier("claude_code") == "tool"


def test_uses_claude_code_detects_nested_providers():
    class Config(BaseModel):
        planner: dict

    assert uses_claude_code(Config(planner={"provider": "claude_code", "model": "m"}))
    assert not uses_claude_code(Config(planner={"provider": "google", "model": "m"}))


@pytest.fixture
def fake_cli(tmp_path: Path) -> Path:
    """A fake `claude` that records its argv/env and prints a canned result."""
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{tmp_path}/argv"\n'
        f'env > "{tmp_path}/env"\n'
        f'cat > "{tmp_path}/stdin"\n'
        f"cat <<'EOF'\n{_stdout({**RESULT_EVENT, 'structured_output': {'verdict': 'pass'}})}EOF\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


class Verdict(BaseModel):
    verdict: str


@pytest.mark.asyncio
async def test_structured_output_runs_cli_without_api_key(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    model = ChatClaudeCode(cli_path=str(fake_cli))

    result = await model.with_structured_output(Verdict).ainvoke([HumanMessage(content="judge")])

    assert result == Verdict(verdict="pass")
    assert "ANTHROPIC_API_KEY" not in (fake_cli.parent / "env").read_text()
    argv = (fake_cli.parent / "argv").read_text().splitlines()
    assert json.loads(argv[argv.index("--json-schema") + 1])["required"] == ["verdict"]
    assert json.loads((fake_cli.parent / "stdin").read_text())["message"]["role"] == "user"


def test_sync_invoke_uses_same_pipeline(fake_cli):
    model = ChatClaudeCode(cli_path=str(fake_cli))
    result = model.with_structured_output(Verdict).invoke([HumanMessage(content="judge")])
    assert result.verdict == "pass"
