"""
Checks the two things this service used to get wrong: its shared secret was a
literal committed to the repository, and its health endpoint answered 200 even
when the recommender had failed to load and every real endpoint was returning 503.

Run with:  python -m unittest test_auth_and_health -v
The Recommender itself is stubbed, so this needs no dataset and no model build.
"""
import importlib
import os
import sys
import types
import unittest

VALID_KEY = "x" * 32
PUBLISHED_KEY = "carshowcase_internal_2026"  # the value that used to be hardcoded


def _install_recommender_stub(should_fail=False):
    """Stands in for recommender.py so the tests never build a real model."""
    module = types.ModuleType("recommender")

    class StubRecommender:
        def __init__(self):
            if should_fail:
                raise RuntimeError("dataset missing")

        def get_similar_cars(self, model):
            return [{"brand": "porsche", "model": "macan"}]

        def get_personalized_ranking(self, prefs):
            return [{"brand": "porsche", "model": "macan"}]

        def update_feedback(self, car_id, positive):
            return None

    module.Recommender = StubRecommender
    sys.modules["recommender"] = module


def _load_app(key, recommender_fails=False):
    if key is None:
        os.environ.pop("ML_SERVICE_KEY", None)
    else:
        os.environ["ML_SERVICE_KEY"] = key
    _install_recommender_stub(should_fail=recommender_fails)
    sys.modules.pop("app", None)
    return importlib.import_module("app")


class StartupFailsClosed(unittest.TestCase):
    def test_refuses_to_start_without_a_key(self):
        with self.assertRaises(SystemExit):
            _load_app(None)

    def test_refuses_to_start_with_an_empty_key(self):
        with self.assertRaises(SystemExit):
            _load_app("   ")

    def test_refuses_the_old_hardcoded_key_as_too_short(self):
        # Anyone with the repo has this value, so it must not be usable.
        with self.assertRaises(SystemExit):
            _load_app(PUBLISHED_KEY)

    def test_starts_with_a_long_enough_key(self):
        app_module = _load_app(VALID_KEY)
        self.assertEqual(app_module.API_KEY, VALID_KEY)


class KeyIsEnforced(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.app_module = _load_app(VALID_KEY)
        # Entered as a context manager so the startup event runs and the recommender
        # is actually loaded; otherwise every endpoint answers 503 for the wrong reason.
        self.client = TestClient(self.app_module.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_request_without_a_key_is_rejected(self):
        self.assertEqual(self.client.get("/recommend/similar/macan").status_code, 403)

    def test_request_with_the_wrong_key_is_rejected(self):
        response = self.client.get("/recommend/similar/macan", headers={"X-ML-KEY": "wrong"})
        self.assertEqual(response.status_code, 403)

    def test_request_with_the_old_hardcoded_key_is_rejected(self):
        response = self.client.get("/recommend/similar/macan", headers={"X-ML-KEY": PUBLISHED_KEY})
        self.assertEqual(response.status_code, 403)

    def test_request_with_the_right_key_is_served(self):
        response = self.client.get("/recommend/similar/macan", headers={"X-ML-KEY": VALID_KEY})
        self.assertEqual(response.status_code, 200)

    def test_feedback_endpoint_also_requires_the_key(self):
        body = {"car_id": "porsche_macan", "action": "like"}
        self.assertEqual(self.client.post("/recommend/feedback", json=body).status_code, 403)
        ok = self.client.post("/recommend/feedback", json=body, headers={"X-ML-KEY": VALID_KEY})
        self.assertEqual(ok.status_code, 200)


class HealthTellsTheTruth(unittest.TestCase):
    def test_healthy_when_the_recommender_loaded(self):
        from fastapi.testclient import TestClient
        app_module = _load_app(VALID_KEY)
        with TestClient(app_module.app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_unhealthy_when_the_recommender_failed_to_load(self):
        from fastapi.testclient import TestClient
        app_module = _load_app(VALID_KEY, recommender_fails=True)
        with TestClient(app_module.app) as client:
            health = client.get("/health")
            real_endpoint = client.get("/recommend/similar/macan",
                                       headers={"X-ML-KEY": VALID_KEY})
        # The point: these two must agree. The endpoint 503s, so health must not say ok.
        self.assertEqual(real_endpoint.status_code, 503)
        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["status"], "unavailable")

    def test_health_needs_no_key(self):
        # Monitors should not have to hold the shared secret.
        from fastapi.testclient import TestClient
        app_module = _load_app(VALID_KEY)
        with TestClient(app_module.app) as client:
            self.assertEqual(client.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
