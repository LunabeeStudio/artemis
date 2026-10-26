# Copyright 2026 Google LLC
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

import asyncio
from artemis.agents.planner.planner import run_async_planner_validation


from pathlib import Path


class DummyDataEngine:
    def get_agent_friendly_steps(self):
        return []


class DummyCtx:
    data_engine = DummyDataEngine()
    project_dir = str(Path(__file__).resolve().parents[2])


import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from artemis.agents.planner.planner import ValidationResult


@pytest.mark.asyncio
async def test_planner_validation():
    content_before = "- [/] Click the button\n- [ ] Complete task"
    content_after = (
        "- [/] Click the button\n- [ ] Complete task\n- [ ] New subgoal appended at bottom"
    )
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        return_value=ValidationResult(is_approved=True, feedback="Valid plan modification")
    )
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("artemis.agents.planner.planner.get_llm", return_value=mock_llm) as get_llm:
        res = await run_async_planner_validation(
            ctx=DummyCtx(),
            initial_goal="Do the task",
            content_before=content_before,
            content_after=content_after,
            operator_raw_thinking="I need to add a new subgoal",
            operator_native_thinking="Thinking...",
        )
    assert res is not None
    assert res.get("status") == "success"
    # The review runs on the lightweight judge node, never the Planner model.
    assert get_llm.call_args.kwargs.get("name") == "planner_validation"


def _llm_config(nodes: dict | None = None):
    from artemis.config.llm import LLMConfig, _expand_default_into_nodes

    default = {
        "provider": "google",
        "model": "gemini-3.8-flash",
        "fallback": {"provider": "google", "model": "gemini-3.7-flash"},
    }
    return LLMConfig.model_validate(
        _expand_default_into_nodes({"default": default, "nodes": nodes or {}})
    )


def test_planner_validation_node_defaults_to_lightweight_judge():
    """Unconfigured planner_validation resolves to the same flash-lite default
    as the pixel safety net (cheap, temperature 0)."""
    llm_cfg = _llm_config()
    node = llm_cfg.get_agent("planner_validation")
    safety_net = llm_cfg.get_agent("validator_pixel_safety_net")
    assert node.model == safety_net.model
    assert node.provider == safety_net.provider
    assert node.temperature == 0.0
    assert "lite" in node.model


def test_configured_planner_validation_node_inherits_default():
    """A planner_validation entry in 'nodes' is honored and inherits 'default'."""
    llm_cfg = _llm_config({"planner_validation": {"model": "claude-haiku-5-5"}})
    node = llm_cfg.get_agent("planner_validation")
    assert (node.provider, node.model) == ("google", "claude-haiku-5-5")
    assert node.fallback.model == "gemini-3.7-flash"
    assert llm_cfg.get_agent("validator_pixel_safety_net").model == "gemini-3.5-flash-lite"


if __name__ == "__main__":
    asyncio.run(test_planner_validation())
