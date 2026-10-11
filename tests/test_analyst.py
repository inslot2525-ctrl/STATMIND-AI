"""AI analyst: Claude when configured, built-in rules otherwise. No test touches the network."""
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

import config
from services import ai_analyst

CONTEXT = {
    "dataset": {"rows": 569, "columns": 31, "domain": "research"},
    "mlResult": {
        "algorithm": "logistic_regression",
        "task_type": "classification",
        "metrics": {"accuracy": 0.9737, "f1_weighted": 0.9736},
        "baseline_metrics": {"accuracy": 0.6316},
        "feature_importance": [{"feature": "worst texture", "importance": 0.02, "std": 0.004}],
    },
}


class FakeAnthropic:
    """Stands in for anthropic.Anthropic and records the request."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.request = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def with_options(self, **_options):
        return self

    def _create(self, **request):
        self.request = request
        if self.error:
            raise self.error
        return self.response


def _reply(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
    )


@pytest.fixture
def llm(monkeypatch):
    """Turn the LLM path on with a fake client; returns a function to install it."""
    monkeypatch.setattr(config, "ANALYST_LLM_ENABLED", True)

    def install(**kwargs):
        fake = FakeAnthropic(**kwargs)
        monkeypatch.setattr(anthropic, "Anthropic", lambda: fake)
        return fake

    return install


def test_rules_answer_when_no_credentials_are_configured(client, alice):
    r = client.post(
        "/api/chat-analyst", headers=alice, json={"question": "What is the accuracy?", "context": CONTEXT}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analyst_mode"] == "rule_based_contextual"
    assert any("0.9737" in line for line in body["answer"])
    assert "No Anthropic credentials" in body["note"]


def test_empty_question_is_rejected(client, alice):
    r = client.post("/api/chat-analyst", headers=alice, json={"question": "  ", "context": {}})
    assert r.status_code == 400


def test_llm_answer_is_grounded_in_the_snapshot(llm):
    fake = llm(response=_reply("It beats the baseline by 34 points.\n\nThe test split is small."))

    result = ai_analyst.answer_data_scientist_question(
        "Can I trust it?",
        CONTEXT,
        history=[
            {"role": "assistant", "content": "orphan reply with no question before it"},
            {"role": "user", "content": "What did I train?"},
            {"role": "assistant", "content": "A logistic regression."},
        ],
    )

    assert result["analyst_mode"] == "llm"
    assert result["answer"] == ["It beats the baseline by 34 points.", "The test split is small."]

    request = fake.request
    assert request["model"] == config.ANALYST_MODEL
    assert request["system"] == ai_analyst.ANALYST_SYSTEM_PROMPT
    # History is replayed, starting from a user turn; the new question comes last
    assert [m["role"] for m in request["messages"]] == ["user", "assistant", "user"]

    final = request["messages"][-1]["content"]
    assert final.endswith("Can I trust it?")
    snapshot = json.loads(final.split("<statmind_snapshot>\n")[1].split("\n</statmind_snapshot>")[0])
    assert snapshot["ml_metrics"]["accuracy"] == 0.9737
    assert snapshot["model"]["baseline_metrics"] == {"accuracy": 0.6316}
    assert snapshot["model"]["feature_importance"][0]["feature"] == "worst texture"


def _status_error(cls, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize("error, expected_note", [
    (_status_error(anthropic.AuthenticationError, 401), "API key was rejected"),
    (_status_error(anthropic.RateLimitError, 429), "rate limited"),
    (_status_error(anthropic.InternalServerError, 500), "returned an error"),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com")), "could not be reached"),
])
def test_api_failures_fall_back_to_rules(llm, error, expected_note):
    llm(error=error)

    result = ai_analyst.answer_data_scientist_question("What is the accuracy?", CONTEXT)

    assert result["analyst_mode"] == "rule_based_contextual"
    assert expected_note in result["note"]
    assert any("0.9737" in line for line in result["answer"])


def test_refusal_falls_back_to_rules(llm):
    llm(response=_reply("", stop_reason="refusal"))

    result = ai_analyst.answer_data_scientist_question("What is the accuracy?", CONTEXT)

    assert result["analyst_mode"] == "rule_based_contextual"
    assert "declined" in result["note"]
