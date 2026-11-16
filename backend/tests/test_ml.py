"""Tests for ML serving, health, demo seeding and collector uploads."""
import numpy as np
import pytest

from app.models.structure import Bay
from app.routers import admin
from app.services.seed_service import ensure_demo_structure, DEMO_STRUCTURE_ID


def still_window(accel, n=20, seed=0):
    """A phone held still: constant gravity vector + small sensor noise."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n, 6))
    w[:, :3] = np.array(accel) + rng.normal(0, 0.1, (n, 3))
    w[:, 3:] = rng.normal(0, 0.5, (n, 3))
    return w.round(2).tolist()


class TestHealth:
    def test_health(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok", "model_loaded": True}


class TestPredict:
    def test_flat_on_table_is_stationary(self, client):
        res = client.post("/api/ml/predict", json={"samples": still_window([0, 0, 9.8])})
        assert res.status_code == 200
        data = res.json()
        assert data["activity"] == "STATIONARY"
        assert set(data["probabilities"]) == {"STAIRS", "STATIONARY", "WALKING"}
        assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
        assert data["inference_ms"] >= 0

    @pytest.mark.parametrize("accel", [[0, 6.9, 6.9], [0, 9.2, 3.4], [0, -6.9, -6.9]])
    def test_held_in_front_is_stationary(self, client, accel):
        """The posture used on the Navigate screen (and iOS's flipped signs) must not read as WALKING."""
        res = client.post("/api/ml/predict", json={"samples": still_window(accel)})
        assert res.json()["activity"] == "STATIONARY"

    def test_wrong_shape_rejected(self, client):
        res = client.post("/api/ml/predict", json={"samples": [[0, 0, 9.8]] * 20})
        assert res.status_code == 422
        assert "Expected 20 samples × 6 values" in res.json()["detail"]

    def test_too_few_samples_rejected(self, client):
        res = client.post("/api/ml/predict", json={"samples": still_window([0, 0, 9.8], n=10)})
        assert res.status_code == 422

    def test_activity_shows_on_admin_sessions(self, seeded_client):
        sid = seeded_client.post("/api/sessions/entry", json={"structure_id": "test_struct"}).json()["session_id"]
        seeded_client.post("/api/ml/predict", json={"samples": still_window([0, 6.9, 6.9]), "session_id": sid})

        sessions = seeded_client.get("/api/admin/sessions").json()["sessions"]
        activity = next(s for s in sessions if s["id"] == sid)["activity"]
        assert activity["activity"] == "STATIONARY"
        assert "updated_at" in activity

    def test_model_info(self, client):
        client.post("/api/ml/predict", json={"samples": still_window([0, 0, 9.8])})
        data = client.get("/api/ml/model").json()
        assert data["loaded"] is True
        assert data["card"]["classes"] == ["STAIRS", "STATIONARY", "WALKING"]
        assert data["card"]["evaluation"]["method"] == "leave-one-session-out"
        assert data["serving_latency"]["count"] >= 1


class TestSeed:
    def test_ensure_demo_structure_is_idempotent(self, db):
        assert ensure_demo_structure(db) is True
        assert ensure_demo_structure(db) is False
        bays = db.query(Bay).filter(Bay.structure_id == DEMO_STRUCTURE_ID).all()
        assert len(bays) == 20
        assert sum(b.bay_type == "ev" for b in bays) == 4

    def test_seed_endpoint(self, client):
        assert client.post("/api/admin/seed").json()["seeded"] is True
        second = client.post("/api/admin/seed").json()
        assert second["seeded"] is False and second["bays"] == 20


class TestTrainingData:
    CSV = (
        f"{admin.CSV_HEADER}\n"
        "1785102382375,6.4,-6.2,3,5.3,2.8,0.3,246.8,-39.8,-74,WALKING\n"
        "1785102382475,6.1,-6.0,3,5.1,2.5,0.2,246.0,-39.0,-74,WALKING\n"
        "1785102382575,0,0.1,9.8,0,0,0,285.2,-3.2,-3.5,STATIONARY\n"
    )

    @pytest.fixture(autouse=True)
    def tmp_training_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(admin, "TRAINING_DIR", tmp_path)

    def test_upload_and_list(self, client):
        res = client.post(
            "/api/admin/training-data?filename=navin session 1.csv",
            content=self.CSV, headers={"Content-Type": "text/csv"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["rows"] == 3
        assert data["labels"] == {"WALKING": 2, "STATIONARY": 1}
        assert data["filename"].endswith("navin_session_1.csv")

        files = client.get("/api/admin/training-data").json()["files"]
        assert [f["filename"] for f in files] == [data["filename"]]
        assert files[0]["rows"] == 3

        download = client.get(f"/api/admin/training-data/{data['filename']}")
        assert download.status_code == 200
        assert download.text.startswith(admin.CSV_HEADER)

    def test_rejects_non_collector_csv(self, client):
        res = client.post("/api/admin/training-data", content="a,b,c\n1,2,3\n",
                          headers={"Content-Type": "text/csv"})
        assert res.status_code == 400


def walking_window(n=20, seed=0):
    """Crude gait: ~2 Hz vertical bounce + sway on top of gravity, with wrist rotation."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 10
    w = np.zeros((n, 6))
    w[:, 0] = 1.5 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.3, n)
    w[:, 1] = 6.9 + 1.0 * np.sin(2 * np.pi * 2.0 * t + 1) + rng.normal(0, 0.3, n)
    w[:, 2] = 6.9 + 3.5 * np.sin(2 * np.pi * 2.0 * t) + rng.normal(0, 0.3, n)
    w[:, 3:] = rng.normal(0, 25, (n, 3))
    return w.round(2).tolist()


class TestSmoothing:
    def test_single_outlier_window_is_outvoted(self, client):
        """Two still windows then one moving window: the session answer stays STATIONARY."""
        moving = walking_window()
        raw = client.post("/api/ml/predict", json={"samples": moving}).json()
        assert raw["activity"] != "STATIONARY" and raw["smoothed_over"] == 1

        for seed in (1, 2):
            client.post("/api/ml/predict", json={"samples": still_window([0, 6.9, 6.9], seed=seed), "session_id": "s-smooth"})
        res = client.post("/api/ml/predict", json={"samples": moving, "session_id": "s-smooth"}).json()
        assert res["smoothed_over"] == 3
        assert res["raw_activity"] == raw["activity"]
        assert res["activity"] == "STATIONARY"

    def test_sessions_do_not_share_history(self, client):
        client.post("/api/ml/predict", json={"samples": still_window([0, 0, 9.8]), "session_id": "s-a"})
        res = client.post("/api/ml/predict", json={"samples": still_window([0, 0, 9.8]), "session_id": "s-b"}).json()
        assert res["smoothed_over"] == 1

    def test_stale_history_is_dropped(self, client, monkeypatch):
        from app.services import ml_service
        client.post("/api/ml/predict", json={"samples": still_window([0, 0, 9.8]), "session_id": "s-stale"})
        real = ml_service.time.monotonic
        monkeypatch.setattr(ml_service.time, "monotonic", lambda: real() + 60)
        res = client.post("/api/ml/predict", json={"samples": still_window([0, 0, 9.8]), "session_id": "s-stale"}).json()
        assert res["smoothed_over"] == 1


class TestCors:
    def test_vercel_origin_allowed(self, client):
        res = client.get("/api/health", headers={"Origin": "https://navin-black.vercel.app"})
        assert res.headers["access-control-allow-origin"] == "https://navin-black.vercel.app"

    def test_unknown_origin_not_allowed(self, client):
        res = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in res.headers
