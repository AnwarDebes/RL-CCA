"""Training logger - structured JSONL output and live status dashboard.

All training observability flows through this module. Writes are append-only
and atomic (temp-file + rename for status.txt) so a crash mid-write doesn't
corrupt logs.

Output layout (relative to run_dir):

  config_snapshot.json          # written once at run start
  iter_metrics.jsonl            # 1 line per iteration
  self_play_summaries.jsonl     # 1 line per generated game
  server_eval_summary.jsonl     # 1 line per (iter, N) eval cell
  rule_alignment.jsonl          # 1 line per alignment check
  status.txt                    # human-readable dashboard, regenerated each iter
  trajectories/                 # sampled compressed trajectories
  policy_value_probes/          # network outputs on fixed test states
  server_eval/                  # raw per-game logs from teacher's server
  plots/                        # auto-generated PNGs (created by plot_training.py)
  stdout.log                    # raw training process stdout/stderr
"""

import gzip
import io
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


def _atomic_write(path: str, content: str) -> None:
    """Write content to path atomically (temp + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single JSON record as a line; flushes immediately."""
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def make_run_dir(base: str = "/home/coder/nexus/training_logs",
                 run_name: Optional[str] = None) -> str:
    """Create and return a fresh run directory."""
    if run_name is None:
        run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, run_name)
    for sub in ("trajectories", "policy_value_probes", "server_eval", "plots"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    # also maintain a `latest` symlink for convenience
    latest = os.path.join(base, "latest")
    try:
        if os.path.islink(latest):
            os.unlink(latest)
        elif os.path.exists(latest):
            os.rename(latest, latest + ".old")
        os.symlink(run_name, latest)
    except OSError:
        pass  # not fatal if symlink can't be made
    return run_dir


class TrainingLogger:
    """Structured logger for one training run.

    Usage:
        logger = TrainingLogger(run_dir)
        logger.write_config_snapshot({...})
        for it in range(num_iters):
            ...
            logger.log_iter({"iter": it, ...})
            logger.regenerate_status()  # cheap, refresh dashboard
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        for sub in ("trajectories", "policy_value_probes", "server_eval", "plots"):
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
        self.iter_path = os.path.join(run_dir, "iter_metrics.jsonl")
        self.game_path = os.path.join(run_dir, "self_play_summaries.jsonl")
        self.eval_path = os.path.join(run_dir, "server_eval_summary.jsonl")
        self.align_path = os.path.join(run_dir, "rule_alignment.jsonl")
        self.status_path = os.path.join(run_dir, "status.txt")
        self.config_path = os.path.join(run_dir, "config_snapshot.json")
        self.start_ts = time.time()

    # ── one-time setup ────────────────────────────────────────────

    def write_config_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Save a snapshot of Config + git rev + run params at start."""
        snapshot = dict(snapshot)
        snapshot.setdefault("started_at", datetime.now().isoformat())
        snapshot.setdefault("run_dir", self.run_dir)
        with open(self.config_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

    # ── always-on streams ────────────────────────────────────────

    def log_iter(self, metrics: Dict[str, Any]) -> None:
        rec = dict(metrics)
        rec.setdefault("ts", datetime.now().isoformat())
        rec.setdefault("wall_sec_total", time.time() - self.start_ts)
        _append_jsonl(self.iter_path, rec)

    def log_self_play_game(self, summary: Dict[str, Any]) -> None:
        _append_jsonl(self.game_path, summary)

    def log_server_eval(self, summary: Dict[str, Any]) -> None:
        rec = dict(summary)
        rec.setdefault("ts", datetime.now().isoformat())
        _append_jsonl(self.eval_path, rec)

    def log_rule_alignment(self, summary: Dict[str, Any]) -> None:
        rec = dict(summary)
        rec.setdefault("ts", datetime.now().isoformat())
        _append_jsonl(self.align_path, rec)

    # ── sampled artifacts ────────────────────────────────────────

    def save_trajectory(self, iter_n: int, N: int, trajectory: List[Dict],
                        suffix: str = "") -> str:
        """Save a self-play trajectory as a compressed pickle. Returns the path."""
        name = f"iter_{iter_n:04d}_N{N}"
        if suffix:
            name = name + "_" + suffix
        name += ".pkl.gz"
        path = os.path.join(self.run_dir, "trajectories", name)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            pickle.dump(trajectory, gz, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path, "wb") as f:
            f.write(buf.getvalue())
        return path

    def save_policy_value_probes(self, iter_n: int, probes: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, "policy_value_probes",
                            f"iter_{iter_n:04d}_probes.json")
        with open(path, "w") as f:
            json.dump(probes, f, default=lambda o: float(o)
                      if hasattr(o, "__float__") else str(o))
        return path

    def server_eval_dir(self, iter_n: int) -> str:
        d = os.path.join(self.run_dir, "server_eval", f"iter_{iter_n:04d}")
        os.makedirs(d, exist_ok=True)
        return d

    # ── live dashboard ───────────────────────────────────────────

    def _read_jsonl(self, path: str, tail: int = -1) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records[-tail:] if tail > 0 else records

    def regenerate_status(self) -> None:
        """Render `status.txt` from the JSONL streams."""
        iters = self._read_jsonl(self.iter_path)
        evals = self._read_jsonl(self.eval_path)
        aligns = self._read_jsonl(self.align_path, tail=1)

        lines = []
        run_name = os.path.basename(self.run_dir)
        lines.append(f"NEXUS Training - {run_name}")
        if iters:
            last = iters[-1]
            lines.append(f"Iter:           {last.get('iter', '?')}")
            lines.append(f"Wall time:      {self._fmt_secs(last.get('wall_sec_total', 0))}")
        else:
            lines.append("Iter:           (no iters logged yet)")

        # Last 5 iters table
        if iters:
            recent = iters[-5:]
            lines.append("")
            lines.append("─── Recent iters ───")
            lines.append(f"{'iter':>5} {'loss':>6} {'pol':>6} {'val':>6} "
                         f"{'lr':>8} {'sec':>5} {'games':>5}  N-mix")
            for r in recent:
                tr = r.get("training", {})
                sp = r.get("self_play", {})
                by_n = sp.get("by_N", {}) or {}
                n_mix = " ".join(f"{k}:{v}" for k, v in sorted(by_n.items()))
                lines.append(
                    f"{r.get('iter','?'):>5} "
                    f"{tr.get('loss_total',0):>6.3f} "
                    f"{tr.get('loss_policy',0):>6.3f} "
                    f"{tr.get('loss_value',0):>6.3f} "
                    f"{tr.get('lr',0):>8.5f} "
                    f"{int(r.get('wall_sec_self_play',0) + r.get('wall_sec_train',0)):>5} "
                    f"{sp.get('games_generated',0):>5}  {n_mix}"
                )

        # Last 3 server evals - pivot to (iter, N) -> mean_score table
        if evals:
            lines.append("")
            lines.append("─── Last 3 server evals ───")
            # group by iter
            by_iter: Dict[int, Dict[int, float]] = {}
            for e in evals:
                it = e.get("iter")
                N = e.get("N")
                ms = e.get("mean_final_score", 0.0)
                if it is None or N is None:
                    continue
                by_iter.setdefault(int(it), {})[int(N)] = float(ms)
            iter_keys = sorted(by_iter.keys())[-3:]
            header = f"{'iter':>5}  " + "  ".join(f"N={n:<5}" for n in (2,3,4,5,6)) + "   mean"
            lines.append(header)
            for it in iter_keys:
                row = by_iter[it]
                vals = [row.get(n, None) for n in (2,3,4,5,6)]
                cells = "  ".join(f"{v:>6.1f}" if v is not None else f"{'-':>6}"
                                  for v in vals)
                present = [v for v in vals if v is not None]
                mean = sum(present) / len(present) if present else 0.0
                lines.append(f"{it:>5}  {cells}   {mean:>6.1f}")

        # Rule alignment status
        if aligns:
            a = aligns[-1]
            ok = a.get("alignment_ok", False)
            lines.append("")
            lines.append("─── Rule alignment ───")
            lines.append(f"Last check: iter {a.get('iter','?')} - "
                         f"{'OK' if ok else 'FAIL'} "
                         f"({a.get('moves_replayed','?')} moves replayed)")

        # Auto-pause flag
        flag = os.path.join(self.run_dir, "RULE_ALIGNMENT_FAILED")
        if os.path.exists(flag):
            lines.append("")
            lines.append("⚠ TRAINING PAUSED: rule alignment failed. See "
                         "rule_alignment.jsonl and remove flag to resume.")

        _atomic_write(self.status_path, "\n".join(lines) + "\n")

    @staticmethod
    def _fmt_secs(s: float) -> str:
        s = int(s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}h {m:02d}m {sec:02d}s"

    # ── halt / resume ────────────────────────────────────────────

    def write_halt_flag(self, name: str, reason: str) -> str:
        """Write a flag file that the trainer checks each iter to halt."""
        path = os.path.join(self.run_dir, name)
        _atomic_write(path, json.dumps({
            "reason": reason,
            "ts": datetime.now().isoformat(),
        }, indent=2) + "\n")
        return path

    def has_halt_flag(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.run_dir, name))
