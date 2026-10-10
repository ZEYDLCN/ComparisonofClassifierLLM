"""HTTP katmanı: yeniden deneme, geri çekilme ve uçtan uca gecikme ölçümü."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import requests

from .core import ProviderError

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_local = threading.local()


def _session() -> requests.Session:
    # İş parçacığı başına oturum: bağlantı yeniden kullanılır (gerçek istemci davranışı).
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


@dataclass
class HttpResult:
    data: dict
    latency_ms: float  # başarılı SON denemenin süresi
    retries: int


def _body_excerpt(resp: requests.Response, n: int = 300) -> str:
    try:
        return resp.text[:n].replace("\n", " ")
    except Exception:  # pragma: no cover
        return ""


def post_json(url: str, *, headers: dict, payload: dict, timeout: float, max_retries: int) -> HttpResult:
    attempt = 0
    last_err = ""
    while True:
        t0 = time.perf_counter()
        try:
            resp = _session().post(url, headers={"Content-Type": "application/json", **headers},
                                   json=payload, timeout=timeout)
            dt = (time.perf_counter() - t0) * 1000
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = f"ağ hatası: {type(e).__name__}"
            resp = None
            dt = (time.perf_counter() - t0) * 1000
        if resp is not None:
            if resp.status_code < 300:
                try:
                    return HttpResult(resp.json(), dt, attempt)
                except ValueError as e:
                    raise ProviderError(f"JSON olmayan yanıt: {_body_excerpt(resp)}") from e
            last_err = f"HTTP {resp.status_code}: {_body_excerpt(resp)}"
            if resp.status_code not in RETRY_STATUS:
                raise ProviderError(last_err)
        if attempt >= max_retries:
            raise ProviderError(f"{last_err} ({attempt} yeniden deneme sonrası)")
        wait = min(30.0, 1.5 ** attempt + random.random())
        if resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra and ra.replace(".", "", 1).isdigit():
                wait = min(60.0, float(ra))
        time.sleep(wait)
        attempt += 1
