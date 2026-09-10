"""
Human-readable console output for the whole pipeline.

Everything the user sees goes through here, so the wording, alignment and
colours stay consistent across every script. The rules:

  * one idea per line, plain language, no internal jargon;
  * every line says what happened, not what function ran;
  * numbers always come with their units and their meaning;
  * anything the user may need to *act* on is a WARNING or a HINT, never
    buried in normal output.

Symbols degrade to ASCII automatically on terminals that cannot print them,
and colours turn themselves off when the output is piped to a file or when
NO_COLOR is set.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

# ----------------------------------------------------------------------
# Capabilities of this terminal
# ----------------------------------------------------------------------

_COLOR = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM", "") != "dumb"
)


def _unicode_ok() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


_UNI = _unicode_ok()

_SYM = {
    "ok": ("✓", "OK "),
    "warn": ("!", "!  "),
    "err": ("✗", "X  "),
    "arrow": ("→", "->"),
    "bullet": ("•", "-"),
    "dot_full": ("█", "#"),
    "dot_empty": ("░", "."),
    "rule": ("─", "-"),
    "heavy": ("═", "="),
    "middot": ("·", "-"),
}


def sym(name: str) -> str:
    uni, ascii_ = _SYM[name]
    return uni if _UNI else ascii_


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def cyan(t: str) -> str:
    return _c("36", t)


def _width(default: int = 72) -> int:
    try:
        return max(48, min(shutil.get_terminal_size().columns, 100))
    except Exception:
        return default


# ----------------------------------------------------------------------
# Verbosity
# ----------------------------------------------------------------------

_QUIET = False
_VERBOSE = False
_WARNINGS: list[str] = []


def configure(quiet: bool = False, verbose: bool = False) -> None:
    """Set the global verbosity once, at the top of a script."""
    global _QUIET, _VERBOSE
    _QUIET, _VERBOSE = quiet, verbose


def _emit(line: str = "") -> None:
    if not _QUIET:
        print(line, flush=True)


# ----------------------------------------------------------------------
# Structure: title banners, numbered steps, sections
# ----------------------------------------------------------------------

def title(text: str, subtitle: str = "") -> None:
    """The banner every script opens with."""
    w = _width()
    bar = sym("heavy") * w
    _emit()
    _emit(cyan(bar))
    _emit(bold(f"  GOALBALL THROWER IDENTIFICATION {sym('middot')} {text}"))
    if subtitle:
        _emit(dim(f"  {subtitle}"))
    _emit(cyan(bar))
    _emit()


_step_state = {"i": 0, "n": 0, "t0": 0.0}


def step(index: int, total: int, text: str) -> None:
    """`STEP 2/6  Marking the court` — the user always knows where they are."""
    _step_state.update(i=index, n=total, t0=time.time())
    _emit(bold(f"STEP {index}/{total}  {text}"))


def step_done(note: str = "") -> None:
    dt = time.time() - _step_state["t0"]
    tail = f"  {sym('middot')} {note}" if note else ""
    _emit(dim(f"          finished in {human_duration(dt)}{tail}"))
    _emit()


def section(text: str) -> None:
    _emit()
    _emit(bold(text))


def rule() -> None:
    _emit(dim(sym("rule") * _width()))


# ----------------------------------------------------------------------
# Lines
# ----------------------------------------------------------------------

def info(text: str) -> None:
    _emit(f"  {text}")


def ok(text: str) -> None:
    _emit(f"  {green(sym('ok'))} {text}")


def warn(text: str, hint: str = "") -> None:
    """Something the user should know about, but the run continues."""
    if text in _WARNINGS:
        return                      # say each thing once, however many callers
    _WARNINGS.append(text)
    _emit(f"  {yellow(sym('warn'))} {yellow('WARNING')}  {text}")
    if hint:
        _emit(f"      {dim('what to do:')} {hint}")


def error(text: str, hint: str = "") -> None:
    _emit(f"  {red(sym('err'))} {red('PROBLEM')}  {text}")
    if hint:
        _emit(f"      {dim('what to do:')} {hint}")


def fail(text: str, hint: str = "") -> "NoReturn":  # type: ignore[valid-type]
    """Print a problem the run cannot recover from and stop."""
    error(text, hint)
    _emit()
    sys.exit(1)


def hint(text: str) -> None:
    _emit(f"      {dim('tip:')} {dim(text)}")


def detail(text: str) -> None:
    """Only shown with --verbose: internals, per-frame noise, tuning values."""
    if _VERBOSE:
        _emit(dim(f"      {text}"))


def kv(label: str, value, width: int = 22) -> None:
    """Aligned `label   value` line — the workhorse for run settings."""
    _emit(f"    {label.ljust(width)} {value}")


def bullet(text: str) -> None:
    _emit(f"    {sym('bullet')} {text}")


def blank() -> None:
    _emit()


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------

def human_duration(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def bar(value: float, width: int = 10) -> str:
    """A tiny 0..1 meter, so confidence is readable at a glance."""
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    body = sym("dot_full") * filled + sym("dot_empty") * (width - filled)
    if value >= 0.7:
        return green(body)
    if value >= 0.4:
        return yellow(body)
    return red(body)


def percent(part: float, whole: float) -> str:
    return "n/a" if not whole else f"{100.0 * part / whole:.1f}%"


def table(headers: list[str], rows: list[list], indent: str = "    ") -> None:
    """Small aligned table for accuracy / summary reports."""
    cells = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in cells:
        for i, c in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(c))
    _emit(indent + bold("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))))
    _emit(indent + dim("  ".join(sym("rule") * w for w in widths)))
    for r in cells:
        _emit(indent + "  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def progress(done: int, total: int, message: str = "") -> None:
    """Single rewritten line for long loops (never spams the scrollback)."""
    if _QUIET or not sys.stdout.isatty():
        return
    total = max(1, total)
    pct = 100.0 * done / total
    line = f"  {bar(done / total, 16)} {done}/{total} ({pct:3.0f}%) {message}"
    sys.stdout.write("\r" + line[: _width() - 1].ljust(_width() - 1))
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\r" + " " * (_width() - 1) + "\r")
        sys.stdout.flush()


def closing(output_lines: list[tuple[str, str]] = ()) -> None:
    """The last thing a script prints: what was written and what to do next."""
    if output_lines:
        section("FILES WRITTEN")
        for label, path in output_lines:
            kv(label, path)
    if _WARNINGS:
        section(f"{len(_WARNINGS)} WARNING(S) DURING THIS RUN")
        for w in _WARNINGS:
            bullet(w)
    _emit()


class muted:
    """
    Swallow a third-party library's own printing.

    Loading the detectors and the appearance model produces pages of framework
    chatter that is meaningless to the user and hides the pipeline's own
    output. It is kept when --verbose is on.

        with logs.muted():
            model = SomeLibrary()
    """

    def __enter__(self):
        if _VERBOSE:
            self._files = None
            return self
        import contextlib
        import io
        self._buffer = io.StringIO()
        self._files = (contextlib.redirect_stdout(self._buffer),
                       contextlib.redirect_stderr(self._buffer))
        for f in self._files:
            f.__enter__()
        return self

    def __exit__(self, *exc):
        if self._files:
            for f in reversed(self._files):
                f.__exit__(*exc)
        return False


def next_steps(lines: list[str]) -> None:
    section("WHAT TO DO NEXT")
    for line in lines:
        _emit(f"    {sym('arrow')} {line}")
    _emit()
