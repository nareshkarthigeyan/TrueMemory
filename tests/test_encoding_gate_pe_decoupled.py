"""Test that prediction error is decoupled from novelty (issue #110)."""

from unittest.mock import patch, MagicMock

import numpy as np


class MockMemoryWithScore:
    """Returns results with a specific score."""
    def __init__(self, score: float, content: str = "existing"):
        self._score = score
        self._content = content
    def search(self, query, **kwargs):
        if self._score > 0:
            return [{"content": self._content, "score": self._score}]
        return []
    def search_vectors(self, query, limit=5):
        return self.search(query)


def _make_mock_model():
    """Mock embedding model that produces distinct 256-d vectors per input."""
    mock = MagicMock()
    def _encode(texts, **kw):
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % (2**31))
            vecs.append(rng.rand(256).astype(np.float32))
        return np.array(vecs)
    mock.encode = _encode
    return mock


def test_pe_not_clamped_at_high_novelty():
    """PE should NOT return a fixed 0.30 when novelty is high."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithScore(score=0.0))

    with patch("truememory.ingest.encoding_gate.EncodingGate._compute_prediction_error", return_value=0.0):
        decision_noise = gate.evaluate("ok", "")

    with patch("truememory.ingest.encoding_gate.EncodingGate._compute_prediction_error", return_value=0.45):
        decision_signal = gate.evaluate("I just got promoted to VP and my salary is now $350,000", "personal")

    assert decision_noise.prediction_error != 0.30 or decision_signal.prediction_error != 0.30, (
        f"PE should not be clamped to 0.30 for high-novelty messages. "
        f"noise PE={decision_noise.prediction_error}, signal PE={decision_signal.prediction_error}"
    )


def test_pe_not_forced_zero_at_low_novelty():
    """PE should NOT be forced to 0.0 when novelty is very low."""
    from truememory.ingest.encoding_gate import EncodingGate

    mock_model = _make_mock_model()
    gate = EncodingGate(memory=MockMemoryWithScore(score=0.95, content="I work at Stripe"))

    with patch("truememory.vector_search.get_model", return_value=mock_model):
        decision = gate.evaluate("I switched from Stripe to Anthropic", "personal")

    assert decision.prediction_error > 0.0, (
        f"PE should not be forced to 0.0 for near-duplicates that contain updates. "
        f"Got PE={decision.prediction_error}"
    )
