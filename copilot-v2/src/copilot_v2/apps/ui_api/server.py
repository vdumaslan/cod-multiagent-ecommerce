from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from copilot_v2.agents.advocate_agent import run_advocate
from copilot_v2.agents.critic_agent import run_critic
from copilot_v2.agents.inventory_agent import run_inventory
from copilot_v2.agents.judge_agent import run_judge
from copilot_v2.agents.pricing_agent import run_pricing
from copilot_v2.agents.retrieval_agent import run_retrieval
from copilot_v2.agents.sentiment_agent import run_sentiment
from copilot_v2.apps.ui_api.plan_store import save_plan_to_bigquery


class Handler(BaseHTTPRequestHandler):
    server_version = "copilot_v2_ui_api/1"

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True, "service": "ui_api"})
            return
        self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
            path = self.path
            if path == "/ui/retrieval":
                self._json(200, {"ok": True, "output": run_retrieval(user_query=str(body.get("user_query", "")))})
                return
            if path == "/ui/sentiment":
                self._json(
                    200,
                    {
                        "ok": True,
                        "output": run_sentiment(
                            user_query=str(body.get("user_query", "")),
                            retrieval_output=body.get("retrieval_output", {}),
                        ),
                    },
                )
                return
            if path == "/ui/pricing":
                self._json(
                    200,
                    {
                        "ok": True,
                        "output": run_pricing(
                            user_query=str(body.get("user_query", "")),
                            retrieval_output=body.get("retrieval_output", {}),
                        ),
                    },
                )
                return
            if path == "/ui/inventory":
                self._json(
                    200,
                    {
                        "ok": True,
                        "output": run_inventory(
                            user_query=str(body.get("user_query", "")),
                            retrieval_output=body.get("retrieval_output", {}),
                        ),
                    },
                )
                return
            if path == "/ui/debate/advocate":
                self._json(200, {"ok": True, "output": run_advocate(**body)})
                return
            if path == "/ui/debate/critic":
                self._json(200, {"ok": True, "output": run_critic(**body)})
                return
            if path == "/ui/debate/judge":
                self._json(200, {"ok": True, "output": run_judge(**body)})
                return
            if path == "/ui/save_plan":
                self._json(
                    200,
                    {
                        "ok": True,
                        "output": save_plan_to_bigquery(
                            plan_id=str(body.get("plan_id", "")),
                            title=str(body.get("title", "")),
                        ),
                    },
                )
                return
            self._json(404, {"ok": False, "error": "not_found"})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"ok": False, "error": type(exc).__name__, "message": str(exc)})


def run_server(host: str = "127.0.0.1", port: int = 8010) -> None:
    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    print(json.dumps({"ok": True, "host": host, "port": int(port), "service": "ui_api"}, indent=2))
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()

