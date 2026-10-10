"""Yerel sahte API sunucusu: gerçek anahtar olmadan tüm sağlayıcı protokollerini test eder.

Dokümanlardan alınan şemaları taklit eder:
  POST /v1/systemone, /v1/decisions   System One (Jev, Mercury Decide, d1, Instinct)
  POST /oai/v1/decisions              OpenAI Decisions
  POST /v1/chat/completions           OpenAI uyumlu sohbet
  POST /v1beta/models/<m>:generateContent   Gemini
  POST /v1/messages                   Anthropic Messages
Metindeki özel işaretler hata durumlarını tetikler: __500__ (kalıcı 500), __flaky__ (bir kez 429), __badlabel__ (geçersiz etiket).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RULES = [
    ("kargo_teslimat", ["kargo", "kurye", "teslimat", "paket", "gümrük"]),
    ("iade_degisim", ["iade", "değişim", "iptal", "cayma"]),
    ("fatura_odeme", ["fatura", "ödeme", "çekim", "taksit", "kart"]),
    ("hesap_guvenlik", ["şifre", "hesab", "yetkisiz", "doğrulama"]),
    ("teknik_destek", ["uygulama", "hata", "çöküyor", "açılmıyor", "site"]),
    ("satis_fiyat", ["fiyat", "kampanya", "stok", "indirim"]),
    ("sikayet_oneri", ["şikayet", "teşekkür", "memnun", "öneri"]),
]


def classify(text: str, labels: list[str]) -> tuple[str, dict[str, float]]:
    low = text.lower()
    pick = "diger" if "diger" in labels else labels[0]
    for lab, kws in RULES:
        if lab in labels and any(k in low for k in kws):
            pick = lab
            break
    if "__badlabel__" in low:
        pick = "olmayan_etiket"
    rest = [l for l in labels if l != pick]
    probs = {pick: 0.85} if pick in labels else {}
    for l in rest:
        probs[l] = (1 - 0.85) / max(len(rest), 1) if pick in labels else 1 / len(labels)
    return pick, probs


class _Handler(BaseHTTPRequestHandler):
    server_version = "mock/1"
    flaky_seen: set = set()
    log: list = []  # (path, headers, body)

    def log_message(self, *a):  # sessiz
        pass

    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _Handler.log.append((self.path, dict(self.headers), body))
        auth = self.headers.get("Authorization", "") or self.headers.get("x-api-key", "") or self.headers.get("x-goog-api-key", "")
        if "good-key" not in auth:
            return self._send(401, {"error": "bad key"})
        path = self.path.split("?")[0]
        text = _extract_text(path, body)
        if "__500__" in text:
            return self._send(500, {"error": "boom"})
        if "__flaky__" in text and text not in _Handler.flaky_seen:
            _Handler.flaky_seen.add(text)
            return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
        if path in ("/v1/systemone", "/v1/decisions"):
            q = body["questions"]["routing"]
            labels = list(q["criteria"])
            pick, probs = classify(text, labels)
            return self._send(200, {"model": body["model"], "answers": {"routing": {
                "type": "choice", "choice": pick, "probabilities": probs, "confidence": max(probs.values())}},
                "usage": {"input_tokens": 100 + len(text) // 4, "output_tokens": 0}})
        if path == "/oai/v1/decisions":
            q = body["questions"][0]
            labels = [c["value"] for c in q["choices"]]
            pick, probs = classify(text, labels)
            return self._send(200, {"answers": [{
                "type": "choice", "name": q["name"], "choice": pick,
                "probabilities": [{"value": k, "probability": v} for k, v in probs.items()],
                "confidence": max(probs.values())}]})
        if path == "/v1/chat/completions":
            labels = _labels_from_system(body["messages"][0]["content"])
            pick, probs = classify(text, labels)
            return self._send(200, {"choices": [{"message": {"content": json.dumps(
                {"label": pick, "confidence": 0.9})}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 12}})
        if path.startswith("/v1beta/models/"):
            labels = body["generationConfig"]["responseSchema"]["properties"]["label"]["enum"]
            pick, probs = classify(text, labels)
            return self._send(200, {"candidates": [{"content": {"parts": [
                {"text": "(düşünce)", "thought": True},
                {"text": json.dumps({"label": pick, "confidence": 0.8})}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 250, "candidatesTokenCount": 10, "thoughtsTokenCount": 40}})
        if path == "/v1/messages":
            labels = body["extra_body"]["output_config"]["format"]["schema"]["properties"]["label"]["enum"] \
                if "extra_body" in body else body["output_config"]["format"]["schema"]["properties"]["label"]["enum"]
            pick, _ = classify(text, labels)
            return self._send(200, {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                                    "content": [{"type": "text", "text": json.dumps({"label": pick, "confidence": 0.7})}],
                                    "stop_reason": "end_turn", "stop_sequence": None,
                                    "usage": {"input_tokens": 280, "output_tokens": 15}})
        self._send(404, {"error": f"yol yok: {path}"})


def _labels_from_system(system: str) -> list[str]:
    labels = []
    for line in system.splitlines():
        if line.startswith("- ") and ":" in line:
            labels.append(line[2:].split(":", 1)[0].strip())
    return labels


def _extract_text(path: str, body: dict) -> str:
    if path in ("/v1/systemone", "/v1/decisions"):
        s = body.get("state")
        return s["message"] if isinstance(s, dict) else str(s)
    if path == "/oai/v1/decisions":
        return str(body.get("input"))
    if path == "/v1/chat/completions" or path == "/v1/messages":
        msgs = body.get("messages", [])
        last = msgs[-1]["content"] if msgs else ""
        return last if isinstance(last, str) else json.dumps(last)
    if path.startswith("/v1beta/models/"):
        return body["contents"][0]["parts"][0]["text"]
    return ""


def start_mock() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
