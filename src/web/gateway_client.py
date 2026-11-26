"""
src/web/gateway_client.py
=========================
Resilient HTTP Client with Circuit Breaker for API Gateway <-> AI Inference Server.
Guarantees graceful degradation when the AI Inference container restarts or is overloaded:
- CLOSED: Normal forwarding with bounded timeout.
- OPEN: Fails fast for `recovery_timeout` seconds so Gateway + PostgreSQL UI stay responsive.
- HALF_OPEN: Probes AI Inference once cooldown expires and auto-recovers cleanly.
"""

from __future__ import annotations
import threading
import time
from typing import Any, Dict, Optional

import httpx


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the AI Inference circuit breaker is OPEN."""


class InferenceCircuitBreakerClient:
    """Thread-safe Circuit Breaker HTTP client targeting the `ai-inference` service."""

    def __init__(
        self,
        base_url: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 8.0,
        timeout: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = float(recovery_timeout)
        self.timeout = float(timeout)

        self._lock = threading.Lock()
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._half_open_in_flight = False
        self.consecutive_failures = 0
        self.opened_at = 0.0
        self.failures_total = 0
        self.requests_total = 0
        self.last_latency_ms = 0.0
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=60, keepalive_expiry=30.0)
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, limits=limits)

    def _before_request(self):
        with self._lock:
            self.requests_total += 1
            if self.state == "OPEN":
                if (time.monotonic() - self.opened_at) >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self._half_open_in_flight = True
                else:
                    raise CircuitBreakerOpenError("AI Inference circuit breaker is OPEN")
            elif self.state == "HALF_OPEN":
                if self._half_open_in_flight:
                    raise CircuitBreakerOpenError("AI Inference circuit breaker HALF_OPEN probe in flight")
                self._half_open_in_flight = True

    def _record_success(self, latency_ms: float):
        with self._lock:
            self.consecutive_failures = 0
            self.state = "CLOSED"
            self._half_open_in_flight = False
            self.last_latency_ms = round(latency_ms, 2)

    def _record_failure(self):
        with self._lock:
            self.failures_total += 1
            self.consecutive_failures += 1
            if self.state == "HALF_OPEN" or self.consecutive_failures >= self.failure_threshold:
                self.state = "OPEN"
                self.opened_at = time.monotonic()
            self._half_open_in_flight = False

    def request_json(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Any = None,
    ) -> Dict[str, Any]:
        self._before_request()
        t0 = time.perf_counter()
        try:
            resp = self._client.request(method, path, json=json_body, params=params, files=files)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code >= 500:
                self._record_failure()
                raise RuntimeError(f"AI Inference returned HTTP {resp.status_code}")
            if resp.status_code >= 400:
                self._record_success(latency_ms)
                try:
                    if resp.headers.get("content-type", "").startswith("application/json"):
                        detail = resp.json().get("detail", resp.text)
                    else:
                        detail = resp.text
                except Exception:
                    detail = resp.text
                return {"_http_error": resp.status_code, "detail": detail}
            payload = resp.json()
            self._record_success(latency_ms)
            return payload
        except CircuitBreakerOpenError:
            raise
        except Exception as exc:
            self._record_failure()
            raise RuntimeError(f"AI Inference unavailable: {exc}") from exc

    def close(self):
        self._client.close()

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "is_open": self.state == "OPEN",
                "consecutive_failures": self.consecutive_failures,
                "failures_total": self.failures_total,
                "requests_total": self.requests_total,
                "last_latency_ms": self.last_latency_ms,
                "target_url": self.base_url,
            }
