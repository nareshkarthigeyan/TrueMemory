"""Regression test for H10 #496: update() holds write lock during model inference.

The fix pre-computes embeddings outside the lock (like add() does),
then only holds _write_lock for the DB writes.
"""

import inspect
import unittest


class TestUpdateLockScope(unittest.TestCase):
    """Verify update() does not hold _write_lock during embedding."""

    def test_update_computes_embedding_outside_lock(self):
        """The embed call must happen BEFORE 'with self._write_lock'."""
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.update)
        lock_pos = source.find("self._write_lock")
        self.assertGreater(lock_pos, -1, "update() should use _write_lock")
        embed_call = source.find("embed_single")
        if embed_call != -1:
            self.fail(
                "update() should NOT call embed_single() inside the lock — "
                "it should pre-compute embeddings like add() does"
            )

    def test_update_pre_computes_like_add(self):
        """update() should use _encode_with_mps_fallback before the lock."""
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.update)
        self.assertIn("_encode_with_mps_fallback", source,
                       "update() should pre-compute embeddings with _encode_with_mps_fallback")


if __name__ == "__main__":
    unittest.main()
