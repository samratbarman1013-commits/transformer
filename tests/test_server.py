"""API server tests using the demo backend (no torch required)."""
import json

from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"]
    assert "web_search" in body["tools"]


def test_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"]


def test_chat_completion():
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello there"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "hello there" in body["choices"][0]["message"]["content"]


def test_chat_completion_stream():
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[len("data: "):]))
    assert chunks, "no SSE chunks received"
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    streamed = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert streamed.strip(), "streamed content was empty"
