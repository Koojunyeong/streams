import builtins
import importlib
import os
import sys
import unittest
from contextlib import contextmanager


@contextmanager
def torch_import_blocked():
    original_import = builtins.__import__
    original_torch_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "torch" or name.startswith("torch.")
    }
    for name in original_torch_modules:
        sys.modules.pop(name, None)

    def blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch intentionally blocked for fallback test")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = blocked_import
    try:
        yield
    finally:
        builtins.__import__ = original_import
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)
        sys.modules.update(original_torch_modules)


def import_server_without_torch():
    os.environ["SUPABASE_URL"] = ""
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = ""
    os.environ["SUPABASE_KEY"] = ""
    sys.modules.pop("server", None)
    with torch_import_blocked():
        return importlib.import_module("server")


class FallbackApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = import_server_without_torch()
        cls.client = cls.server.app.test_client()

    def test_health_reports_fallback_mode_when_torch_is_unavailable(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {
                "model_loaded": False,
                "ai_mode": "fallback",
                "ai_ready": True,
                "device": "cpu",
            },
            {
                key: response.get_json()[key]
                for key in ("model_loaded", "ai_mode", "ai_ready", "device")
            },
        )
        self.assertFalse(self.server.TORCH_AVAILABLE)
        self.assertFalse(self.server.loaded)

    def test_ai_move_returns_valid_fallback_response(self):
        payload = {
            "board": [0] * 20,
            "current_num": 15,
            "deck": [15, 1, 30, 11, 20, 5, 25, 12, 18, 8, 22, 2, 28, 13, 17, 9, 24, 3, 27, 10],
            "current_index": 0,
        }

        response = self.client.post("/api/ai_move", json=payload)
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(data["action"], int)
        self.assertGreaterEqual(data["action"], 0)
        self.assertLess(data["action"], 20)
        self.assertEqual(data["q_values"], [0] * 20)
        self.assertEqual(len(data["prob_mask"]), 20)

    def test_ai_move_rejects_invalid_payload(self):
        response = self.client.post("/api/ai_move", json={"board": [0] * 20})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["status"], "error")

    def test_probability_mask_allows_equal_numbers(self):
        board = [0] * 20
        board[18] = 15
        deck = [15, 15] + list(range(1, 11)) + [11, 12, 13, 14, 16, 17, 18, 19]

        mask = self.server.get_prob_mask(board, 15, deck, 1)

        self.assertEqual(mask[19], 1.0)


if __name__ == "__main__":
    unittest.main()
