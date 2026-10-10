import json

import pytest
from fastapi.testclient import TestClient

from backend import app as lumina
from backend import safety


class Forbidden:
    """Stands in for memory / Gemini on the crisis path: touching it fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"{name} was called on the crisis path")


class Recorder:
    def __init__(self, **returns):
        self.calls, self.returns = [], returns

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.returns.get(name)
        return call


class FakeModel:
    def __init__(self, risk="safe", crash=False):
        self.risk, self.crash = risk, crash

    def predict(self, text):
        if self.crash:
            raise RuntimeError("model down")
        return {"emotion": "sadness", "emotion_confidence": 0.9, "intensity": 0.7, "risk": self.risk, "risk_confidence": 0.8}


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(lumina, "DB", tmp_path / "test.db")
    monkeypatch.setattr(lumina, "analyzer", None)
    lumina.init_db()


def test_critical_short_circuits(monkeypatch):
    monkeypatch.setattr(lumina, "memory", Forbidden())
    monkeypatch.setattr(lumina, "llm", Forbidden())
    reply, a = lumina.run_pipeline(1, "i want to kill myself", genz=True)
    assert reply == safety.CRISIS_MESSAGE  # fixed wording, Gen-Z tone ignored
    assert a["level"] == safety.CRITICAL and a["llm"] == "skipped" and a["memory"] == []


def test_ml_can_escalate_to_critical(monkeypatch):
    monkeypatch.setattr(lumina, "analyzer", FakeModel(risk="critical"))
    monkeypatch.setattr(lumina, "memory", Forbidden())
    monkeypatch.setattr(lumina, "llm", Forbidden())
    reply, a = lumina.run_pipeline(1, "i just feel like a ghost lately", genz=False)
    assert a["rule"]["level"] == safety.SAFE and a["level"] == safety.CRITICAL
    assert reply == safety.CRISIS_MESSAGE


def test_ml_cannot_downgrade_rule_gate(monkeypatch):
    monkeypatch.setattr(lumina, "analyzer", FakeModel(risk="safe"))
    monkeypatch.setattr(lumina, "llm", Forbidden())
    monkeypatch.setattr(lumina, "memory", Forbidden())
    _, a = lumina.run_pipeline(1, "kms", genz=False)
    assert a["level"] == safety.CRITICAL


def test_model_crash_still_runs_rule_gate(monkeypatch):
    monkeypatch.setattr(lumina, "analyzer", FakeModel(crash=True))
    monkeypatch.setattr(lumina, "llm", Forbidden())
    monkeypatch.setattr(lumina, "memory", Forbidden())
    reply, a = lumina.run_pipeline(1, "i want to die", genz=False)
    assert a["ml"] is None and a["level"] == safety.CRITICAL and reply == safety.CRISIS_MESSAGE


def test_safe_message_uses_memory_and_therapist(monkeypatch):
    monkeypatch.setattr(lumina, "analyzer", FakeModel())
    mem = Recorder(search=[{"summary": "The user is stressed about exams.", "date": "2026-10-01", "distance": 0.3}])
    llm = Recorder(reply="That sounds like a lot.")
    monkeypatch.setattr(lumina, "memory", mem)
    monkeypatch.setattr(lumina, "llm", llm)
    reply, a = lumina.run_pipeline(1, "exams again ugh", genz=True)
    assert reply == "That sounds like a lot." and a["llm"] == "ok"
    name, args, _ = llm.calls[0]
    assert name == "reply" and args[1]["emotion"] == "sadness" and args[3] == mem.returns["search"] and args[5] is True


def test_next_message_after_crisis_checks_in(monkeypatch):
    monkeypatch.setattr(lumina, "memory", Recorder(search=[]))
    llm = Recorder(reply="Are you okay right now?")
    monkeypatch.setattr(lumina, "llm", llm)
    lumina.q("INSERT INTO messages (session_id, role, text, analysis) VALUES (1, 'user', 'i want to die', ?)",
             json.dumps({"level": "critical"}))
    _, a = lumina.run_pipeline(1, "sorry, i'm here", genz=True)
    assert a["after_crisis"] and llm.calls[-1][2]["after_crisis"] is True
    lumina.q("INSERT INTO messages (session_id, role, text, analysis) VALUES (1, 'user', 'sorry, i''m here', ?)",
             json.dumps({"level": "safe"}))
    _, a = lumina.run_pipeline(1, "yeah i'm okay now", genz=True)
    assert not a["after_crisis"] and llm.calls[-1][2]["after_crisis"] is False


def test_prompt_after_crisis_drops_slang(monkeypatch):
    seen = {}
    monkeypatch.setattr(lumina.llm, "_generate", lambda system, contents: seen.setdefault("system", system))
    lumina.llm.reply("hi", None, "safe", [], [], genz=True, after_crisis=True)
    assert lumina.llm.AFTER_CRISIS in seen["system"] and lumina.llm.GENZ not in seen["system"]
    seen.clear()
    lumina.llm.reply("hi", None, "safe", [], [], genz=True)
    assert lumina.llm.GENZ in seen["system"] and lumina.llm.AFTER_CRISIS not in seen["system"]


def test_gemini_failure_falls_back(monkeypatch):
    monkeypatch.setattr(lumina, "memory", Recorder(search=[]))
    monkeypatch.setattr(lumina.llm, "reply", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429")))
    reply, a = lumina.run_pipeline(1, "rough day", genz=False)
    assert reply == lumina.llm.FALLBACK and a["llm"] == "unavailable"


def test_crisis_text_never_reaches_gemini_history():
    crit = json.dumps({"level": "critical"})
    ok = json.dumps({"level": "safe"})
    for role, text, a in [("user", "i want to die", crit), ("assistant", safety.CRISIS_MESSAGE, crit),
                          ("user", "thanks, a bit better", ok), ("assistant", "I'm glad.", ok)]:
        lumina.q("INSERT INTO messages (session_id, role, text, analysis) VALUES (1, ?, ?, ?)", role, text, a)
    history = lumina.history_for_llm(1)
    texts = " ".join(m["text"] for m in history)
    assert "want to die" not in texts and safety.CRISIS_MESSAGE not in texts
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]


def test_end_to_end_api(monkeypatch):
    monkeypatch.setattr(lumina, "memory", Recorder(search=[]))
    monkeypatch.setattr(lumina.llm, "available", lambda: True)
    monkeypatch.setattr(lumina.llm, "reply", lambda *a, **k: "I hear you.")
    monkeypatch.setattr(lumina.llm, "summarize", lambda t: "The user talked about a rough day.")
    c = TestClient(lumina.app)
    sid = c.post("/api/sessions").json()["id"]
    r = c.post(f"/api/sessions/{sid}/messages", json={"text": "rough day"}).json()
    assert r["reply"] == "I hear you." and "resources" not in r
    r = c.post(f"/api/sessions/{sid}/messages", json={"text": "i want to end my life"}).json()
    assert r["reply"] == safety.CRISIS_MESSAGE and r["resources"]
    assert c.post(f"/api/sessions/{sid}/messages", json={"text": ""}).status_code == 422
    s = c.post(f"/api/sessions/{sid}/end").json()
    assert s["summary"] and json.loads(s["stats"])["highest_level"] == "critical"
    assert c.get(f"/api/sessions/{sid}/messages").json() == []  # raw transcript deleted
    assert c.post(f"/api/sessions/{sid}/messages", json={"text": "hi"}).status_code == 409
