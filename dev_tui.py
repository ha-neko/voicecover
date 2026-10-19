#!/usr/bin/env python3
"""Flicker-free ncurses dashboard for the Voicecover web stack."""

from __future__ import annotations

import curses
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".dev-runtime"
LOG_DIR = RUNTIME / "logs"
DEV_SH = ROOT / "dev.sh"
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
SPARKS = "▁▂▃▄▅▆▇█"
BARS = " ▁▂▃▄▅▆▇█"
WEB_PORT = int(os.environ.get("VOICECOVER_WEB_PORT", "8000"))
LLM_PORT = int(os.environ.get("VOICECOVER_LLM_PORT", "8081"))


@dataclass(frozen=True)
class Service:
    key: str
    label: str
    endpoint: str
    port: int | None


@dataclass
class Snapshot:
    service: Service
    state: str
    pid: int | None
    uptime: str = "—"
    cpu: float = 0.0
    rss_kb: int = 0


@dataclass
class CpuSnapshot:
    total: float
    cores: list[float]
    history: list[float]


@dataclass
class JobSnapshot:
    status: str
    stage: str
    messages: list[str]
    error: str | None = None
    video_url: str | None = None


SERVICES = (
    Service("web", "Voicecover Web", f"http://127.0.0.1:{WEB_PORT}", WEB_PORT),
    Service("llm", "LLM Server", f"http://127.0.0.1:{LLM_PORT}", LLM_PORT),
    Service("tunnel", "HTTPS Tunnel", "waiting for URL", None),
)


# ── low-level helpers ──────────────────────────────────────────────


def read_pid(service: str) -> int | None:
    try:
        fields = (RUNTIME / f"{service}.pid").read_text().split()
        if len(fields) != 2 or not all(value.isdigit() for value in fields):
            return None
        pid, recorded_start = map(int, fields)
        current_start = int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])
        return pid if recorded_start == current_start else None
    except (OSError, ValueError, IndexError):
        return None


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def port_open(port: int) -> bool:
    wanted = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            rows = table.read_text().splitlines()[1:]
        except OSError:
            continue
        if any(
            len(fields := row.split()) > 3
            and fields[1].rsplit(":", 1)[-1] == wanted
            and fields[3] == "0A"
            for row in rows
        ):
            return True
    return False


def clean_line(value: str) -> str:
    value = ANSI_RE.sub("", value).replace("\r", "").replace("\t", "  ")
    return "".join(c if c.isprintable() else " " for c in value)


def tail_lines(path: Path, count: int, offset: int = 0) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 262_144))
            lines = [clean_line(line) for line in handle.read().decode(errors="replace").splitlines()]
    except OSError:
        return []
    end = max(0, len(lines) - offset)
    return lines[max(0, end - count) : end]


def fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def format_kb(value: int) -> str:
    if value >= 1_048_576:
        return f"{value / 1_048_576:.1f}G"
    if value >= 1024:
        return f"{value / 1024:.0f}M"
    return f"{value}K"


def sparkline(values: list[float], width: int) -> str:
    values = values[-width:]
    if not values:
        return " " * width
    ceiling = max(max(values), 1.0)
    return "".join(SPARKS[min(7, int(v / ceiling * 7))] for v in values).rjust(width)


def bar(value: float, width: int) -> str:
    filled = int(min(value / 100.0, 1.0) * width)
    empty = width - filled
    return BARS[8] * filled + BARS[0] * empty


# ── CPU reader ─────────────────────────────────────────────────────


_prev: dict[str, list[int]] | None = None


def read_cpu() -> tuple[float, list[float]]:
    global _prev
    try:
        lines = Path("/proc/stat").read_text().splitlines()
    except OSError:
        return 0.0, []

    current: dict[str, list[int]] = {}
    for line in lines:
        if line.startswith("cpu"):
            parts = line.split()
            key = parts[0]
            vals = [int(v) for v in parts[1:]]
            current[key] = vals

    previous = _prev
    total_cores = len(current) - 1
    cores: list[float] = [0.0] * total_cores

    if previous is not None:
        for idx in range(total_cores):
            key = f"cpu{idx}"
            if key in current and key in previous:
                cur, prv = current[key], previous[key]
                d_idle = cur[3] - prv[3]
                d_total = sum(cur) - sum(prv)
                if d_total > 0:
                    cores[idx] = (1.0 - d_idle / d_total) * 100.0

    total_usage = 0.0
    if previous is not None and "cpu" in current and "cpu" in previous:
        total = current["cpu"]
        d_total = sum(total) - sum(previous["cpu"])
        d_idle = total[3] - previous["cpu"][3]
        if d_total > 0:
            total_usage = (1.0 - d_idle / d_total) * 100.0
    _prev = current
    return total_usage, cores


# ── job poller ─────────────────────────────────────────────────────


_STAGE_RE = [
    ("extract", re.compile(r"extracting|downloading", re.I)),
    ("transcribe", re.compile(r"transcribing", re.I)),
    ("translate", re.compile(r"translat|model", re.I)),
    ("voice", re.compile(r"generating.*(?:voice|utterance)|clone voice", re.I)),
    ("create", re.compile(r"synchroniz|creating dubbed|done:", re.I)),
]
_STAGE_LABELS = ["extract", "transcribe", "translate", "voice", "create"]
_STAGE_ICONS = ["1", "2", "3", "4", "5"]


def poll_jobs() -> list[JobSnapshot]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{WEB_PORT}/api/jobs", timeout=0.3) as resp:
            data = json.loads(resp.read())
    except (OSError, ValueError, json.JSONDecodeError):
        return []

    jobs: list[JobSnapshot] = []
    if isinstance(data, dict) and "jobs" in data:
        items = data["jobs"] if isinstance(data["jobs"], list) else []
    elif isinstance(data, list):
        items = data
    else:
        return []

    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "unknown"))
        progress = item.get("progress", [])
        messages = [str(m) for m in progress] if isinstance(progress, list) else []
        error = str(item.get("error")) if item.get("error") else None
        video_url = str(item.get("video_url")) if item.get("video_url") else None

        # Durable history contains previous attempts. A fresh extraction marks
        # the latest run, so stale "done" messages must not force its stage to
        # CREATE while it is currently cloning voice.
        attempt_start = 0
        for index, message in enumerate(messages):
            if re.search(r"extracting audio|visual-only mode|downloading youtube", message, re.I):
                attempt_start = index
        stage_index = -1
        for msg in messages[attempt_start:]:
            for index, (_, regex) in enumerate(_STAGE_RE):
                if regex.search(msg):
                    stage_index = max(stage_index, index)
        stage = _STAGE_RE[stage_index][0] if stage_index >= 0 else "idle"
        if status == "completed":
            stage = "done"
        elif status == "failed":
            stage = "error"

        jobs.append(JobSnapshot(status=status, stage=stage, messages=messages, error=error, video_url=video_url))
    # Newest first within each status class; active work always stays visible.
    jobs.reverse()
    priority = {"running": 0, "queued": 1, "paused": 2, "interrupted": 3, "failed": 4, "cancelled": 5, "completed": 6}
    jobs.sort(key=lambda job: priority.get(job.status, 7))
    return jobs


# ── dashboard ──────────────────────────────────────────────────────


class Dashboard:
    def __init__(self, screen: curses.window) -> None:
        self.screen = screen
        self.selected = 0
        self.snapshots: list[Snapshot] = []
        self.cpu_history = {service.key: [] for service in SERVICES}
        self.load_history: list[float] = []
        self.cpu = CpuSnapshot(total=0.0, cores=[], history=[])
        self.jobs: list[JobSnapshot] = []
        self.colors: dict[str, int] = {}
        self.message = "ready"
        self.message_error = False
        self.paused = False
        self.help_open = False
        self.detail_open = False
        self.log_offset = 0
        self.log_scroll_left = 0  # scroll for left log panel (web)
        self.log_scroll_right = 0  # scroll for right log panel (LLM)
        self.jobs_scroll = 0  # scroll offset for jobs panel
        self.action: subprocess.Popen[str] | None = None
        self.action_label = ""
        self.action_started = 0.0
        self.last_sample = 0.0
        self.spinner = 0
        self._prev_cpu: tuple[float, list[float]] = (0.0, [])
        self.regions: dict[str, tuple[int, int]] = {}
        self.setup_terminal()

    def setup_terminal(self) -> None:
        self.screen.keypad(True)
        self.screen.timeout(100)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        except curses.error:
            pass
        try:
            curses.mouseinterval(0)
        except curses.error:
            pass
        # Ordinary UI deliberately inherits the terminal foreground/background.
        # Color is reserved for semantic indicators only.
        self.colors = {
            "cyan": 0,
            "blue": 0,
            "muted": curses.A_DIM,
            "text": 0,
            "selected": curses.A_BOLD,
            "green": 0,
            "yellow": 0,
            "red": 0,
            "magenta": 0,
            "bar_ok": 0,
            "bar_mid": 0,
            "bar_hi": 0,
        }
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                return
            semantic = {
                "green": curses.COLOR_GREEN,
                "yellow": curses.COLOR_YELLOW,
                "red": curses.COLOR_RED,
                "magenta": curses.COLOR_MAGENTA,
            }
            for index, (name, foreground) in enumerate(semantic.items(), 1):
                try:
                    curses.init_pair(index, foreground, -1)
                    self.colors[name] = curses.color_pair(index)
                except curses.error:
                    continue
            self.colors["bar_ok"] = self.colors["green"]
            self.colors["bar_mid"] = self.colors["yellow"]
            self.colors["bar_hi"] = self.colors["red"]

    @property
    def tunnel_url(self) -> str:
        for line in reversed(tail_lines(LOG_DIR / "tunnel.log", 100)):
            if match := TUNNEL_RE.search(line):
                return match.group(0)
        return "waiting for URL"

    def process_metrics(self, pids: list[int]) -> dict[int, tuple[str, float, int]]:
        if not pids:
            return {}
        try:
            result = subprocess.run(
                ["ps", "-o", "pid=,etime=,%cpu=,rss=", "-p", ",".join(map(str, pids))],
                capture_output=True, text=True, timeout=0.4, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        metrics: dict[int, tuple[str, float, int]] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            try:
                if len(parts) == 4:
                    metrics[int(parts[0])] = (parts[1], float(parts[2]), int(parts[3]))
            except ValueError:
                continue
        return metrics

    def collect(self) -> None:
        # services
        pids = {s.key: read_pid(s.key) for s in SERVICES}
        metrics = self.process_metrics([pid for pid in pids.values() if process_alive(pid) and pid])
        snapshots: list[Snapshot] = []
        for service in SERVICES:
            pid = pids[service.key]
            managed = process_alive(pid)
            if service.key == "web":
                healthy = port_open(service.port or 0)
            elif service.key == "llm":
                healthy = port_open(service.port or 0)
            else:
                healthy = managed and self.tunnel_url != "waiting for URL" and port_open(WEB_PORT)
            state = "online" if managed and healthy else "degraded" if managed else "external" if healthy else "offline"
            snap = Snapshot(service, state, pid if managed else None)
            if snap.pid in metrics:
                snap.uptime, snap.cpu, snap.rss_kb = metrics[snap.pid]
            history = self.cpu_history[service.key]
            history.append(snap.cpu)
            del history[:-30]
            snapshots.append(snap)
        self.snapshots = snapshots

        # load
        try:
            self.load_history.append(os.getloadavg()[0])
        except OSError:
            self.load_history.append(0.0)
        del self.load_history[:-30]

        # cpu per-core (delta-based)
        self.cpu.total, self.cpu.cores = read_cpu()
        self.cpu.history.append(self.cpu.total)
        del self.cpu.history[:-30]

        # jobs
        self.jobs = poll_jobs()
        self.jobs_scroll = min(self.jobs_scroll, max(0, len(self.jobs) - 1))

        self.last_sample = time.monotonic()

    def memory_summary(self) -> tuple[int, int, int]:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0])
        except (OSError, ValueError, IndexError):
            return 0, 0, 0
        total = values.get("MemTotal", 0)
        used = max(0, total - values.get("MemAvailable", 0))
        return used, total, int(used * 100 / total) if total else 0

    def cpu_bar_color(self, pct: float) -> int:
        if pct >= 85:
            return self.colors["bar_hi"]
        if pct >= 55:
            return self.colors["bar_mid"]
        return self.colors["bar_ok"]

    # ── drawing ────────────────────────────────────────────────────

    def add(self, y: int, x: int, text: str, attr: int = 0, width: int | None = None) -> None:
        height, columns = self.screen.getmaxyx()
        if not (0 <= y < height and 0 <= x < columns):
            return
        limit = max(0, min(columns - x - (1 if y == height - 1 else 0), width or columns))
        try:
            self.screen.addnstr(y, x, text, limit, attr)
        except curses.error:
            pass

    def box(self, y: int, x: int, height: int, width: int, title: str) -> None:
        line_attr = curses.A_DIM
        self.add(y, x, "┌" + "─" * (width - 2) + "┐", line_attr, width)
        for row in range(y + 1, y + height - 1):
            self.add(row, x, "│", line_attr)
            self.add(row, x + width - 1, "│", line_attr)
        self.add(y + height - 1, x, "└" + "─" * (width - 2) + "┘", line_attr, width)
        self.add(y, x + 2, f" {fit(title, width - 6)} ", curses.A_BOLD, width - 4)

    def state_attr(self, state: str) -> int:
        return {
            "online": self.colors["green"],
            "degraded": self.colors["yellow"],
            "external": self.colors["magenta"],
            "offline": self.colors["muted"] | curses.A_DIM,
        }[state]

    def endpoint(self, snapshot: Snapshot) -> str:
        return self.tunnel_url if snapshot.service.key == "tunnel" else snapshot.service.endpoint

    def draw_header(self, width: int) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        title = " VOICECOVER  /  LOCAL CONTROL "
        self.add(0, 0, title, curses.A_BOLD)
        self.add(0, width - len(now) - 1, now, curses.A_DIM)
        core = [item for item in self.snapshots if item.service.key in {"web", "llm"}]
        online = sum(item.state == "online" for item in core)
        used, total, percent = self.memory_summary()
        load = self.load_history[-1] if self.load_history else 0
        indicator = f"● {online}/{len(core)} core"
        indicator_attr = self.colors["green"] if online == len(core) else self.colors["yellow"]
        self.add(1, 1, indicator, indicator_attr | curses.A_BOLD)
        tunnel = next((item for item in self.snapshots if item.service.key == "tunnel"), None)
        summary = (
            f"  load {load:.2f} {sparkline(self.load_history, 10)}  "
            f"ram {format_kb(used)}/{format_kb(total)} {percent}%  "
            f"tunnel {tunnel.state if tunnel else 'unknown'}"
        )
        self.add(1, 2 + len(indicator), fit(summary, width - len(indicator) - 3))
        sel = self.snapshots[self.selected]
        self.add(2, 1, fit(f"selected  {sel.service.label}  ·  {self.endpoint(sel)}", width - 2), curses.A_DIM)

    def draw_services(self, y: int, width: int) -> int:
        height = len(self.snapshots) + 3
        self.box(y, 0, height, width, "services")
        self.add(y + 1, 3, "SERVICE          STATE        PID      UPTIME       CPU      MEM      ENDPOINT", curses.A_DIM, width - 5)
        for index, snap in enumerate(self.snapshots):
            row = y + 2 + index
            selected = index == self.selected
            self.add(row, 1, "›" if selected else " ", curses.A_REVERSE | curses.A_BOLD if selected else 0)
            self.add(row, 3, f"{snap.service.label:<16}", curses.A_BOLD if selected else 0, 16)
            self.add(row, 20, f"● {snap.state.upper():<9}", self.state_attr(snap.state) | curses.A_BOLD, 11)
            self.add(row, 32, f"{snap.pid or '—':<8}", 0, 8)
            self.add(row, 41, f"{snap.uptime:<12}", 0, 12)
            self.add(row, 54, f"{snap.cpu:>5.1f}%", 0, 6)
            self.add(row, 62, f"{format_kb(snap.rss_kb):>7}", 0, 7)
            self.add(row, 71, fit(self.endpoint(snap), width - 73), curses.A_DIM, width - 73)
        return y + height

    def draw_cpu(self, y: int, width: int) -> int:
        cores = self.cpu.cores
        n = len(cores)
        if n == 0:
            return y

        # Keep the dashboard usable on ordinary terminals: show the six busiest
        # cores rather than allowing a 32-core host to consume the whole screen.
        visible = sorted(enumerate(cores), key=lambda item: item[1], reverse=True)[:6]
        bar_width = 12
        col_gap = 2
        core_col_w = bar_width + 16  # "core 00 ████... 42%"
        cols_per_row = max(1, (width - col_gap) // core_col_w)
        rows_needed = (len(visible) + cols_per_row - 1) // cols_per_row

        height = rows_needed + 3
        self.box(y, 0, height, width, f"processing units  ·  {n} cores  ·  busiest six")

        # total CPU sparkline
        spark_w = min(40, width - 20)
        spark = sparkline(self.cpu.history, spark_w)
        self.add(y + 1, 2, f"total {self.cpu.total:>5.1f}%  ", curses.A_BOLD)
        self.add(y + 1, 15, spark, self.cpu_bar_color(self.cpu.total), spark_w)

        # per-core bars
        for position, (idx, pct) in enumerate(visible):
            row = y + 2 + position // cols_per_row
            col_offset = position % cols_per_row
            x = 2 + col_offset * core_col_w
            label = f"core{idx:<2}"
            bar_str = bar(pct, bar_width)
            pct_str = f"{pct:>5.1f}%"
            attr = self.cpu_bar_color(pct)
            self.add(row, x, f"{label} ", curses.A_DIM, 6)
            self.add(row, x + 6, bar_str, attr, bar_width)
            self.add(row, x + 6 + bar_width + 1, pct_str, attr, 7)

        return y + height

    def draw_jobs(self, y: int, width: int, max_height: int) -> int:
        desired = min(14, max(5, len(self.jobs) * 3 + 2))
        height = max(4, min(desired, max_height))
        self.box(y, 0, height, width, "pipeline jobs")

        if not self.jobs:
            self.add(y + 1, 2, "no active dubbing jobs", curses.A_DIM, width - 4)
            return y + height

        visible = self.jobs[self.jobs_scroll:]
        content_h = height - 2

        for jidx, job in enumerate(visible):
            if jidx * 3 >= content_h:
                break
            row = y + 1 + jidx * 3

            # status line
            status_color = {
                "queued": self.colors["yellow"],
                "running": self.colors["green"],
                "completed": self.colors["green"] | curses.A_BOLD,
                "failed": self.colors["red"],
                "paused": self.colors["yellow"],
                "interrupted": self.colors["yellow"],
                "cancelled": curses.A_DIM,
            }.get(job.status, curses.A_DIM)

            stage_idx = (
                len(_STAGE_LABELS)
                if job.status == "completed"
                else _STAGE_LABELS.index(job.stage) if job.stage in _STAGE_LABELS else -1
            )

            stage_line = ""
            for i, label in enumerate(_STAGE_LABELS):
                if i < stage_idx:
                    stage_line += f" {label.upper()} "
                elif i == stage_idx:
                    if job.status == "running":
                        stage_line += f" [{label.upper()}] "
                    elif job.status == "completed":
                        stage_line += f" {label.upper()} "
                    else:
                        stage_line += f" {label} "
                else:
                    stage_line += f" {label} "

            status_str = f"● {job.status.upper()}"
            self.add(row, 2, fit(status_str, 12), status_color, 12)
            self.add(row, 15, fit(stage_line, width - 17), curses.A_BOLD if job.status == "running" else 0, width - 17)

            # show last 2 progress messages
            msgs = job.messages[-2:] if job.messages else []
            if job.error:
                msgs = [f"error: {job.error}"]
            for midx, msg in enumerate(msgs):
                msg_row = row + 1 + midx
                if msg_row >= y + height - 1:
                    break
                attr = self.colors["red"] if job.error and midx == len(msgs) - 1 else curses.A_DIM
                self.add(msg_row, 4, fit(msg, width - 6), attr, width - 6)

            if not msgs:
                self.add(row + 1, 4, "waiting...", curses.A_DIM, width - 6)

        # scroll indicator
        total = len(self.jobs)
        if total > content_h // 3:
            pos = self.jobs_scroll + 1
            self.add(y + height - 2, width - 12, f"{pos}/{total} ↑↓", curses.A_DIM, 10)

        return y + height

    def draw_log_line(self, y: int, x: int, line: str, width: int) -> None:
        """Render terminal-default log text with only a semantic marker colored."""
        lowered = line.lower()
        if re.search(r"\b(error|fatal|critical)\b|traceback", lowered):
            marker, attr = "!", self.colors["red"] | curses.A_BOLD
        elif re.search(r"\bwarn(?:ing)?\b", lowered):
            marker, attr = "!", self.colors["yellow"] | curses.A_BOLD
        else:
            marker, attr = "·", curses.A_DIM
        self.add(y, x, marker, attr, 1)
        self.add(y, x + 2, fit(line, width - 2), 0, width - 2)

    def draw_logs(self, y: int, height: int, width: int) -> None:
        gap = 1
        left_width = (width - gap) // 2
        scroll_states = [self.log_scroll_left, self.log_scroll_right]
        for index, (x, pw) in enumerate(((0, left_width), (left_width + gap, width - left_width - gap))):
            snap = self.snapshots[index]
            self.box(y, x, height, pw, f"{snap.service.label} · {snap.state}")
            max_lines = height - 2
            all_lines = tail_lines(LOG_DIR / f"{snap.service.key}.log", max_lines + scroll_states[index]) or ["no log output"]
            # scroll offset: 0 = newest at bottom, higher = older lines visible
            offset = scroll_states[index]
            if offset > 0:
                end = max(0, len(all_lines) - offset)
                start = max(0, end - max_lines)
                visible = all_lines[start:end]
            else:
                visible = all_lines[-max_lines:]
            for line_idx, line in enumerate(visible):
                self.draw_log_line(y + 1 + line_idx, x + 2, line, pw - 4)
            if offset > 0:
                self.add(y + 1, x + pw - 10, f"↑{offset}", curses.A_REVERSE, 8)

    def draw_detail(self, height: int, width: int) -> None:
        snap = self.snapshots[self.selected]
        y, x, ph, pw = 2, 2, height - 4, width - 4
        self.box(y, x, ph, pw, f"full log · {snap.service.label}")
        count = ph - 3
        lines = tail_lines(LOG_DIR / f"{snap.service.key}.log", count, self.log_offset) or ["no log output"]
        for offset, line in enumerate(lines[-count:]):
            self.draw_log_line(y + 1 + offset, x + 2, line, pw - 4)
        mode = "following" if self.log_offset == 0 else f"-{self.log_offset} lines"
        self.add(y + ph - 2, x + 2, f"{mode} · ↑↓ scroll · End follow · l/Esc close", curses.A_DIM, pw - 4)

    def draw_help(self, height: int, width: int) -> None:
        lines = [
            "↑/↓ or j/k       select service",
            "PgUp/PgDn        scroll all logs + jobs panel",
            "z / x            scroll left log (web) up/down",
            "c / v            scroll right log (LLM) up/down",
            "g / G            scroll to top / bottom",
            "Enter / s        toggle or start selected",
            "X / r            stop or restart selected",
            "a / Z            start / stop complete stack",
            "t                toggle HTTPS tunnel",
            "l / C            full log / clear selected log",
            "p / q            pause telemetry / quit dashboard",
        ]
        pw = min(72, width - 4)
        ph = len(lines) + 4
        py, px = (height - ph) // 2, (width - pw) // 2
        for row in range(py, py + ph):
            self.add(row, px, " " * pw, curses.A_REVERSE, pw)
        self.box(py, px, ph, pw, "keyboard help")
        for index, line in enumerate(lines):
            self.add(py + 2 + index, px + 3, line, curses.A_REVERSE, pw - 6)

    def draw_footer(self, height: int, width: int) -> None:
        if self.action and self.action.poll() is None:
            status = f"{'◐◓◑◒'[self.spinner % 4]} {self.action_label}"
            self.spinner += 1
            attr = self.colors["yellow"]
        else:
            status = "Ⅱ telemetry paused" if self.paused else self.message
            attr = self.colors["red"] if self.message_error else curses.A_DIM
        self.add(height - 2, 1, fit(status, width - 2), attr, width - 2)
        keys = "↑↓ select  enter toggle  s start  X stop  r restart  a start all  Z stop all  t tunnel  l log  p pause  ? help  q quit"
        self.add(height - 1, 0, " " * (width - 1), curses.A_REVERSE, width - 1)
        self.add(height - 1, 1, fit(keys, width - 2), curses.A_REVERSE | curses.A_BOLD, width - 2)

    def render(self) -> None:
        height, width = self.screen.getmaxyx()
        self.screen.erase()
        if height < 24 or width < 78:
            text = f"terminal too small: {width}×{height} · minimum 78×24"
            self.add(height // 2, max(0, (width - len(text)) // 2), text, self.colors["red"] | curses.A_BOLD)
        else:
            self.draw_header(width)
            services_y = 3
            services_bottom = self.draw_services(services_y, width)
            self.regions["services"] = (services_y, services_bottom)
            cpu_bottom = self.draw_cpu(services_bottom, width)
            self.regions["cpu"] = (services_bottom, cpu_bottom)
            jobs_budget = max(4, height - cpu_bottom - 7)
            jobs_bottom = self.draw_jobs(cpu_bottom, width, jobs_budget)
            self.regions["jobs"] = (cpu_bottom, jobs_bottom)
            log_h = height - jobs_bottom - 2
            if log_h >= 4:
                self.draw_logs(jobs_bottom, log_h, width)
                self.regions["logs"] = (jobs_bottom, jobs_bottom + log_h)
            self.draw_footer(height, width)
            if self.detail_open:
                self.draw_detail(height, width)
            if self.help_open:
                self.draw_help(height, width)
        self.screen.noutrefresh()
        curses.doupdate()

    def start_action(self, command: str, target: str) -> None:
        if self.action and self.action.poll() is None:
            self.message = "an action is already running"
            self.message_error = True
            return
        self.action_label = f"{command} {target}"
        self.message_error = False
        try:
            self.action = subprocess.Popen(
                [str(DEV_SH), command, target],
                cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
            self.action_started = time.monotonic()
        except OSError as error:
            self.action = None
            self.message = f"{self.action_label} failed · {error}"
            self.message_error = True

    def poll_action(self) -> None:
        if not self.action or self.action.poll() is None:
            if self.action and time.monotonic() - self.action_started > 90:
                try:
                    os.killpg(self.action.pid, signal.SIGTERM)
                except OSError:
                    pass
            return
        stdout, stderr = self.action.communicate()
        lines = [clean_line(line).strip() for line in stdout.splitlines() if clean_line(line).strip()]
        if self.action.returncode == 0:
            self.message = f"{self.action_label} complete" + (f" · {lines[-1]}" if lines else "")
            self.message_error = False
        else:
            self.message = f"{self.action_label} failed · {clean_line(stderr.strip() or (lines[-1] if lines else 'unknown error'))}"
            self.message_error = True
        self.action = None
        self.last_sample = 0

    def handle_key(self, key: int) -> bool:
        height, width = self.screen.getmaxyx()
        if height < 24 or width < 78:
            return key not in (ord("q"), ord("Q"))
        if key in (ord("q"), ord("Q")):
            if self.detail_open:
                self.detail_open = False
                return True
            if self.help_open:
                self.help_open = False
                return True
            return False
        if self.help_open:
            if key in (27, ord("?")):
                self.help_open = False
            return True
        if self.detail_open:
            if key in (27, ord("l"), ord("L")):
                self.detail_open = False
            elif key in (curses.KEY_UP, ord("k")):
                self.log_offset += 1
            elif key in (curses.KEY_DOWN, ord("j")):
                self.log_offset = max(0, self.log_offset - 1)
            elif key == curses.KEY_PPAGE:
                self.log_offset += max(5, height - 10)
            elif key == curses.KEY_NPAGE:
                self.log_offset = max(0, self.log_offset - max(5, height - 10))
            elif key == curses.KEY_END:
                self.log_offset = 0
            elif key == curses.KEY_MOUSE:
                try:
                    _, mx, my, _, bstate = curses.getmouse()
                    if bstate & curses.BUTTON4_PRESSED:
                        self.log_offset += 3
                    elif bstate & getattr(curses, "BUTTON5_PRESSED", curses.BUTTON2_PRESSED):
                        self.log_offset = max(0, self.log_offset - 3)
                except curses.error:
                    pass
            return True
        # mouse scroll on main dashboard
        if key == curses.KEY_MOUSE:
            try:
                _, mx, my, _, bstate = curses.getmouse()
                wheel_down = getattr(curses, "BUTTON5_PRESSED", curses.BUTTON2_PRESSED)
                direction = 1 if bstate & curses.BUTTON4_PRESSED else (-1 if bstate & wheel_down else 0)
                if direction == 0:
                    return True
                services_top, services_bottom = self.regions.get("services", (3, 9))
                cpu_top, cpu_bottom = self.regions.get("cpu", (services_bottom, services_bottom))
                jobs_top, jobs_bottom = self.regions.get("jobs", (cpu_bottom, cpu_bottom))
                if services_top <= my < services_bottom:
                    # scroll services selection
                    self.selected = (self.selected - direction) % len(SERVICES)
                elif cpu_top <= my < cpu_bottom:
                    pass  # cpu panel, no scroll
                elif jobs_top <= my < jobs_bottom:
                    # scroll jobs panel
                    self.jobs_scroll = max(0, min(max(0, len(self.jobs) - 5), self.jobs_scroll + direction * 3))
                else:
                    # log panels — left or right half
                    if mx < width // 2:
                        self.log_scroll_left = max(0, min(200, self.log_scroll_left + direction * 3))
                    else:
                        self.log_scroll_right = max(0, min(200, self.log_scroll_right + direction * 3))
            except curses.error:
                pass
            return True
        if key in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("K")):
            self.selected = (self.selected - 1) % len(SERVICES)
        elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("J")):
            self.selected = (self.selected + 1) % len(SERVICES)
        elif key in (curses.KEY_PPAGE,):
            self.jobs_scroll = min(max(0, len(self.jobs) - 5), self.jobs_scroll + 3)
            self.log_scroll_left = min(200, self.log_scroll_left + 5)
            self.log_scroll_right = min(200, self.log_scroll_right + 5)
        elif key in (curses.KEY_NPAGE,):
            self.jobs_scroll = max(0, self.jobs_scroll - 3)
            self.log_scroll_left = max(0, self.log_scroll_left - 5)
            self.log_scroll_right = max(0, self.log_scroll_right - 5)
        elif key in (ord("z"),):
            self.log_scroll_left = min(200, self.log_scroll_left + 3)
        elif key in (ord("x"),):
            self.log_scroll_left = max(0, self.log_scroll_left - 3)
        elif key in (ord("c"),):
            self.log_scroll_right = min(200, self.log_scroll_right + 3)
        elif key in (ord("v"),):
            self.log_scroll_right = max(0, self.log_scroll_right - 3)
        elif key in (ord("g"), curses.KEY_HOME):
            self.jobs_scroll = 0
            self.log_scroll_left = 0
            self.log_scroll_right = 0
        elif key in (ord("G"), curses.KEY_END):
            self.jobs_scroll = max(0, len(self.jobs) - 5)
            self.log_scroll_left = 0
            self.log_scroll_right = 0
        elif key in (10, 13, ord(" ")):
            item = self.snapshots[self.selected]
            if item.state == "external" and item.pid is None:
                self.message = f"{item.service.label} is external · adopt or stop it manually"
                self.message_error = True
            else:
                self.start_action("stop" if item.pid else "start", item.service.key)
        elif key in (ord("s"), ord("S")):
            self.start_action("start", SERVICES[self.selected].key)
        elif key in (ord("X"),):
            self.start_action("stop", SERVICES[self.selected].key)
        elif key in (ord("r"), ord("R")):
            self.start_action("restart", SERVICES[self.selected].key)
        elif key in (ord("a"), ord("A")):
            self.start_action("start", "all")
        elif key in (ord("Z"),):
            self.start_action("stop", "all")
        elif key in (ord("t"), ord("T")):
            tunnel_index = next(
                index for index, service in enumerate(SERVICES) if service.key == "tunnel"
            )
            self.selected = tunnel_index
            self.start_action(
                "stop" if self.snapshots[tunnel_index].pid else "start", "tunnel"
            )
        elif key in (ord("l"), ord("L")):
            self.detail_open = True
            self.log_offset = 0
        elif key in (ord("C"),):
            try:
                (LOG_DIR / f"{SERVICES[self.selected].key}.log").write_text("")
                self.message = "log cleared"
            except OSError as error:
                self.message, self.message_error = str(error), True
        elif key in (ord("p"), ord("P")):
            self.paused = not self.paused
        elif key in (ord("?"), curses.KEY_F1):
            self.help_open = True
        return True

    def run(self) -> None:
        self.collect()
        while True:
            self.poll_action()
            if not self.paused and time.monotonic() - self.last_sample >= 0.75:
                self.collect()
            self.render()
            key = self.screen.getch()
            if key != -1 and not self.handle_key(key):
                break


def main(screen: curses.window) -> None:
    Dashboard(screen).run()


if __name__ == "__main__":
    curses.wrapper(main)
