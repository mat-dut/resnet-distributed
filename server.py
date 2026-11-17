# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "uvicorn"]
# ///
from __future__ import annotations

import argparse
import csv
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


class RegisterRequest(BaseModel):
    name: str


class WorkRequest(BaseModel):
    agent_id: int
    run: int = 0
    results: list[dict[str, Any]] = []
    seconds: float = 0.0


@dataclass
class Agent:
    name: str
    last_seen: float
    images: int = 0
    busy_seconds: float = 0.0
    has_chunk: bool = False


class Coordinator:
    """Holds the file list, hands out chunks to agents and collects results."""

    def __init__(self, files: list[str], chunk_size: int, output: Path) -> None:
        self.files = files
        self.chunk_size = chunk_size
        self.output = output
        self.lock = threading.Lock()
        self.agents: dict[int, Agent] = {}
        self.run = 0
        self.state = "waiting"
        self.queue: deque[list[str]] = deque()
        self.in_flight = 0
        self.results: list[dict[str, Any]] = []
        self.t_start = 0.0
        self.t_end = 0.0

    def register(self, name: str) -> int:
        """Add a new agent and return its id."""
        with self.lock:
            agent_id = len(self.agents) + 1
            self.agents[agent_id] = Agent(name=name, last_seen=time.time())
            return agent_id

    def start(self) -> None:
        """Split the file list into chunks and start a new run."""
        with self.lock:
            if self.state == "running":
                return
            self.run += 1
            self.state = "running"
            self.queue = deque(self.files[i:i + self.chunk_size] for i in range(0, len(self.files), self.chunk_size))
            self.in_flight = 0
            self.results = []
            for agent in self.agents.values():
                agent.images, agent.busy_seconds, agent.has_chunk = 0, 0.0, False
            self.t_start = time.perf_counter()

    def work(self, req: WorkRequest) -> dict[str, Any]:
        """Accept results of the previous chunk and give the agent its next chunk."""
        with self.lock:
            agent = self.agents.get(req.agent_id)
            if agent is None:
                raise HTTPException(404, "unknown agent")
            agent.last_seen = time.time()

            if agent.has_chunk and req.run == self.run:
                agent.has_chunk = False
                self.in_flight -= 1
                self.results.extend(req.results)
                agent.images += len(req.results)
                agent.busy_seconds += req.seconds

            files: list[str] = []
            if self.state == "running" and self.queue:
                files = self.queue.popleft()
                agent.has_chunk = True
                self.in_flight += 1
            elif self.state == "running" and self.in_flight == 0:
                self.finish()
            return {"run": self.run, "state": self.state, "files": files}

    def finish(self) -> None:
        """Stop the clock and save results."""
        self.t_end = time.perf_counter()
        self.state = "done"
        save_results(self.results, self.output)
        n, s = len(self.results), self.t_end - self.t_start
        print(f"[server] run {self.run}: {n} images in {s:.2f}s  ->  {n / s:.1f} img/s, saved to {self.output}")

    def status(self) -> dict[str, Any]:
        """Current state for the web page."""
        with self.lock:
            now = time.time()
            end = self.t_end if self.state == "done" else time.perf_counter()
            elapsed = end - self.t_start if self.state != "waiting" else 0.0
            done = len(self.results)
            return {
                "state": self.state,
                "total": len(self.files),
                "done": done,
                "elapsed": elapsed,
                "throughput": done / elapsed if elapsed > 0 else 0.0,
                "agents": [
                    {"id": i, "name": a.name, "online": now - a.last_seen < 10, "images": a.images,
                     "speed": a.images / a.busy_seconds if a.busy_seconds > 0 else 0.0}
                    for i, a in self.agents.items()
                ],
            }


def load_file_list(source: Path) -> list[str]:
    """Read file names from a .txt list or by scanning a folder."""
    if source.is_dir():
        return sorted(f.relative_to(source).as_posix() for f in source.rglob("*") if f.suffix.lower() in IMG_EXTS)
    return [line.strip() for line in source.read_text().splitlines() if line.strip()]


def save_results(results: list[dict[str, Any]], output: Path) -> None:
    """Write results to CSV, one row per image."""
    with output.open("w", newline="") as f:
        w = csv.writer(f)
        k = len(results[0]["predictions"]) if results else 0
        header = ["image"]
        for i in range(1, k + 1):
            header += [f"top{i}_label", f"top{i}_id", f"top{i}_prob"]
        w.writerow(header)
        for r in sorted(results, key=lambda r: r["image"]):
            row: list[Any] = [r["image"]]
            for p in r["predictions"]:
                row += [p["label"], p["class_id"], p["prob"]]
            w.writerow(row)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>ResNet-50</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;background:#fafafa;color:#222}
table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}
button{font-size:18px;padding:8px 28px;cursor:pointer}.bar{height:18px;background:#e4e4e4;border-radius:4px;overflow:hidden}
.fill{height:100%;background:#3b82f6;width:0}.big{font-size:22px;margin:12px 0}.off{color:#aaa}
</style></head><body>
<h1>ResNet-50</h1>
<button id="start" onclick="fetch('/start',{method:'POST'}).then(refresh)">Start</button>
<p class="big" id="summary"></p>
<div class="bar"><div class="fill" id="fill"></div></div>
<table><thead><tr><th>#</th><th>Agent</th><th>Status</th><th>Images</th><th>img/s</th></tr></thead><tbody id="agents"></tbody></table>
<script>
async function refresh(){
  const s = await (await fetch('/status')).json();
  document.getElementById('start').disabled = s.state === 'running';
  document.getElementById('fill').style.width = (100 * s.done / s.total) + '%';
  document.getElementById('summary').textContent =
    `${s.state} | ${s.done}/${s.total} images | ${s.elapsed.toFixed(2)} s | ${s.throughput.toFixed(1)} img/s`;
  document.getElementById('agents').innerHTML = s.agents.map(a =>
    `<tr class="${a.online ? '' : 'off'}"><td>${a.id}</td><td>${a.name}</td><td>${a.online ? 'online' : 'offline'}</td>` +
    `<td>${a.images}</td><td>${a.speed.toFixed(1)}</td></tr>`).join('');
}
refresh(); setInterval(refresh, 1000);
</script></body></html>"""


def create_app(coord: Coordinator) -> FastAPI:
    """HTTP endpoints for agents and the web page."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE

    @app.get("/status")
    def status() -> dict[str, Any]:
        return coord.status()

    @app.post("/start")
    def start() -> dict[str, Any]:
        coord.start()
        return coord.status()

    @app.post("/register")
    def register(req: RegisterRequest) -> dict[str, int]:
        return {"agent_id": coord.register(req.name)}

    @app.post("/work")
    def work(req: WorkRequest) -> dict[str, Any]:
        return coord.work(req)

    return app


def main() -> None:
    """Load the file list and run the coordinator server."""
    p = argparse.ArgumentParser(description="Distributed ResNet-50: coordinator server.")
    p.add_argument("files", type=Path, help="Text file with image names, or a folder with images")
    p.add_argument("--chunk-size", type=int, default=32, help="Images per chunk handed to an agent")
    p.add_argument("-o", "--output", type=Path, default=Path("results_dist.csv"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    files = load_file_list(args.files)
    print(f"[server] {len(files)} images, chunks of {args.chunk_size}")
    coord = Coordinator(files, args.chunk_size, args.output)
    uvicorn.run(create_app(coord), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
