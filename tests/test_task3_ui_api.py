"""Integration tests for Task 3 Review 3 UI/UX & API endpoints."""
import io
import numpy as np
import cv2
from fastapi.testclient import TestClient
from src.web.app import app


def test_dashboard_contains_task3_controls():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "INDUSTRIAL CCTV SURVEILLANCE" in r.text
    assert "Image Audit" in r.text
    assert "Runtime &amp; SLA" in r.text
    assert "imageAuditModal" in r.text
    assert "benchmarkModal" in r.text


def test_predict_image_and_feedback_endpoints(tmp_path, monkeypatch):
    import src.web.app as web_app
    fb_file = tmp_path / "operator_feedback.jsonl"
    monkeypatch.setattr(web_app, "FEEDBACK_LOG_PATH", fb_file)

    client = TestClient(app)
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    r = client.post("/api/predict_image", files={"file": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "inference_ms" in data
    assert data["annotated_image_b64"].startswith("data:image/jpeg;base64,")

    r_fb = client.post(
        "/api/feedback",
        json={"item_id": "test_1", "worker_id": 1, "verdict": "confirm", "source_type": "incident"},
    )
    assert r_fb.status_code == 200
    assert r_fb.json()["status"] == "saved"

    r_list = client.get("/api/feedback")
    assert r_list.status_code == 200
    assert r_list.json()["summary"]["confirmed"] == 1


def test_benchmark_summary_endpoint():
    client = TestClient(app)
    r = client.get("/api/benchmark/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "active_model" in data



def test_prometheus_metrics_and_benchmark_summary():
    from fastapi.testclient import TestClient
    from src.web.app import app

    client = TestClient(app)
    r_prom = client.get("/metrics")
    assert r_prom.status_code == 200
    assert "text/plain" in r_prom.headers.get("content-type", "")
    body = r_prom.text
    for metric_name in (
        "aegis_cctv_fps",
        "aegis_cctv_inference_latency_ms",
        "aegis_cctv_compliance_rate_pct",
        "aegis_cctv_cpu_percent",
        "aegis_cctv_ram_used_mb",
        "aegis_cctv_http_requests_total",
    ):
        assert metric_name in body

    r_bench = client.get("/api/benchmark/summary")
    assert r_bench.status_code == 200
    data = r_bench.json()
    assert "host_telemetry" in data
    assert "optimization_before_after" in data


def test_vectorized_biformer_and_repconv_operators():
    import torch
    import torch.nn as nn
    from custom_ablation_modules import BiFormerBlockLite, RepConv

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)

    # Verify RepConv switch_to_deploy equivalence on CPU/IEEE FP32
    rc = RepConv(32, 32, k=3, s=1).eval()
    for mod in rc.modules():
        if isinstance(mod, nn.BatchNorm2d):
            nn.init.normal_(mod.weight, 1.0, 0.2)
            nn.init.uniform_(mod.running_var, 0.5, 2.0)
    x_rc = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        y_train = rc(x_rc)
        rc.switch_to_deploy()
        y_dep = rc(x_rc)
    assert (y_train - y_dep).abs().max().item() < 3e-4

    # Verify BiFormerBlockLite vectorized forward produces finite outputs cleanly across non-divisible spatial dims
    bf = BiFormerBlockLite(32, num_heads=4, region_size=8, topk=4).eval()
    x_bf = torch.randn(1, 32, 20, 20)
    with torch.no_grad():
        y_bf = bf(x_bf)
    assert y_bf.shape == x_bf.shape
    assert torch.isfinite(y_bf).all()
