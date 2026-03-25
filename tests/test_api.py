import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture
def mock_predictor():
    """Mock predictor to avoid loading a real ONNX model."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "predictions": [{"class": "pizza", "confidence": 0.94}],
        "latency_ms": 12.3,
    }
    return predictor


@pytest.fixture
def client(mock_predictor):
    import src.serving.app as app_module
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as c:
        # Set mock after lifespan runs (lifespan won't find model files)
        app_module.predictor = mock_predictor
        yield c


class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_predict_valid_image(self, client):
        img = Image.new("RGB", (224, 224))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        resp = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert resp.status_code == 200
        assert "predictions" in resp.json()

    def test_predict_invalid_file_type(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422
