"""Tool registry and tool behavior tests (no heavy deps needed)."""
from tools.registry import ToolRegistry, ToolSpec
from tools import code_runner, memory, web_search


def test_register_and_execute():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="echo", description="echo", parameters={"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}), lambda s: s.upper())
    result = reg.execute("echo", {"s": "hi"})
    assert result.ok and result.data == "HI"


def test_unknown_tool_is_a_result_not_a_crash():
    reg = ToolRegistry()
    result = reg.execute("nope", {})
    assert not result.ok and "unknown tool" in result.error


def test_missing_required_argument():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="echo", description="echo", parameters={"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}), lambda s: s)
    result = reg.execute("echo", {})
    assert not result.ok and "required" in result.error


def test_default_registry_has_expected_tools():
    reg = ToolRegistry()
    reg.register(web_search.SPEC, web_search.web_search)
    reg.register(code_runner.SPEC, code_runner.run_python)
    names = [s.name for s in reg.specs()]
    assert {"web_search", "run_python"} <= set(names)


def test_code_runner_runs_code():
    out = code_runner.run_python("print(2 + 2)")
    assert out["ok"] and out["stdout"].strip() == "4"


def test_code_runner_blocks_destructive_patterns():
    out = code_runner.run_python("import os\nos.system('rm -rf /')")
    assert not out["ok"] and "blocked" in out["error"]


def test_code_runner_timeout():
    out = code_runner.run_python("while True: pass", timeout=1)
    assert not out["ok"]


def test_memory_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSFORMER_HOME", str(tmp_path))
    assert memory.memory_get() == {}
    memory.memory_set("preferred_language", "English")
    assert memory.memory_get()["preferred_language"] == "English"


def test_web_search_without_key_is_graceful(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    out = web_search.web_search("anything")
    assert "error" in out and "not configured" in out["error"]
