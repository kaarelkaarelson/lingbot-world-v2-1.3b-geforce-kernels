"""`lingbot play`: a local window on the live model.

Threads: LiveSource's generation and host threads produce frames; a consumer thread moves them
from the source into the playout queue (dropping warm-up frames until the source is ready); the
main thread polls the keyboard at ~60 Hz into `InputState`, presents one frame per paced period
and prints the HUD once a second.

Pacing (the stream's `--surplus rate` policy, lingbot/play/pacer.py): frames arrive in bursts of
4 per latent, the source runs slightly faster than real time (16.2 FPS generated vs 16 played),
so a fixed t0 + k/16 schedule would let the queue, and the key->pixel latency, grow by ~0.2
frames per second. The RateController presents at the measured arrival rate (up to +25 % fast,
nothing dropped) and holds a 2-frame trough. `--surplus wait` keeps playback nominal and throttles
the source to generate just in time instead.

Key->pixel is measured the way the stream's ack does: a keydown edge gets a seq; the input sample
that consumed it echoes (seq, first frame index of the slot it landed in); the sample closes when
that chunk's frame at or after the index is presented.
"""
from __future__ import annotations

import collections
import logging
import os
import statistics
import threading
import time

from .control import InputState
from .pacer import RateController

log = logging.getLogger("lingbot.play")

POLL_HZ = 60
HUD_EVERY = 1.0
# tap schedule for the headless modes (--dry, --headless-seconds): W for 0.4 s every 2.5 s from t = 0.1 s
SCRIPT_TAP_KEY, SCRIPT_TAP_START, SCRIPT_TAP_HOLD, SCRIPT_TAP_PERIOD = "KeyW", 0.1, 0.4, 2.5


class ScriptedKeys:
    """Headless input: the tap schedule above, as a poll() with the display's contract."""

    def __init__(self, t0: float):
        self.t0 = t0

    def poll(self):
        t = time.monotonic() - self.t0
        phase = (t - SCRIPT_TAP_START) % SCRIPT_TAP_PERIOD
        held = {SCRIPT_TAP_KEY} if t >= SCRIPT_TAP_START and phase < SCRIPT_TAP_HOLD else set()
        return held, False, False, 0.0, 0.0


class NullDisplay:
    """No window (headless without pygame): frames are consumed, nothing is drawn."""

    def __init__(self, width, height, title):
        pass

    def poll(self):
        return set(), False, False, 0.0, 0.0

    def present(self, rgb):
        pass

    def set_caption(self, text):
        pass

    def close(self):
        pass


class PygameDisplay:
    KEYS = None  # pygame key -> KeyboardEvent.code name (InputState.KEY_AXES)

    def __init__(self, width, height, title):
        import pygame
        self.pg = pygame
        pygame.init()
        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        k = pygame
        self.KEYS = {k.K_w: "KeyW", k.K_s: "KeyS", k.K_a: "KeyA", k.K_d: "KeyD", k.K_q: "KeyQ", k.K_e: "KeyE",
                     k.K_UP: "ArrowUp", k.K_DOWN: "ArrowDown", k.K_LEFT: "ArrowLeft", k.K_RIGHT: "ArrowRight",
                     k.K_LSHIFT: "ShiftLeft", k.K_RSHIFT: "ShiftRight"}

    def poll(self):
        pg = self.pg
        reset = quit_ = False
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                quit_ = True
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    quit_ = True
                elif ev.key == pg.K_r:
                    reset = True
        pressed = pg.key.get_pressed()
        held = {code for key, code in self.KEYS.items() if pressed[key]}
        dx, dy = pg.mouse.get_rel()
        if not pg.mouse.get_pressed()[0]:  # mouse look while the left button is held
            dx = dy = 0.0
        return held, reset, quit_, float(dx), float(dy)

    def present(self, rgb):
        surf = self.pg.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        self.screen.blit(surf, (0, 0))
        self.pg.display.flip()

    def set_caption(self, text):
        self.pg.display.set_caption(text)

    def close(self):
        self.pg.quit()


class Cv2Display:
    """Fallback without pygame: cv2.imshow + waitKey. OpenCV reports key presses (with the OS's
    auto-repeat), not releases, so a key counts as held for HOLD_S after its last report."""

    HOLD_S = 0.15
    KEYS = {ord("w"): "KeyW", ord("s"): "KeyS", ord("a"): "KeyA", ord("d"): "KeyD", ord("q"): "KeyQ", ord("e"): "KeyE",
            # arrows: Linux (GTK) and macOS key codes of waitKeyEx
            65362: "ArrowUp", 65364: "ArrowDown", 65361: "ArrowLeft", 65363: "ArrowRight",
            63232: "ArrowUp", 63233: "ArrowDown", 63234: "ArrowLeft", 63235: "ArrowRight"}

    def __init__(self, width, height, title):
        import cv2
        self.cv2 = cv2
        self.title = title
        self.last_seen: dict[str, float] = {}
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def poll(self):
        reset = quit_ = False
        now = time.monotonic()
        while True:
            k = self.cv2.waitKeyEx(1)
            if k == -1:
                break
            if k == 27:
                quit_ = True
            elif k in (ord("r"), ord("R")):
                reset = True
            elif k in self.KEYS:
                self.last_seen[self.KEYS[k]] = now
            elif (k & 0xFF) in self.KEYS:
                self.last_seen[self.KEYS[k & 0xFF]] = now
        held = {code for code, t in self.last_seen.items() if now - t < self.HOLD_S}
        return held, reset, quit_, 0.0, 0.0

    def present(self, rgb):
        self.cv2.imshow(self.title, rgb[:, :, ::-1])

    def set_caption(self, text):
        self.cv2.setWindowTitle(self.title, text)

    def close(self):
        self.cv2.destroyAllWindows()


def open_display(width, height, title, headless=False):
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        return PygameDisplay(width, height, title)
    except ImportError as e:
        if headless:
            return NullDisplay(width, height, title)
        log.warning("pygame not installed (%s); using OpenCV's window (press-only keys, held for %.0f ms)", e, Cv2Display.HOLD_S * 1000)
    try:
        return Cv2Display(width, height, title)
    except ImportError as e:
        raise SystemExit(f"no window backend: pip install pygame ({e})") from e


class KeyToPixel:
    """Attribute presented frames to keydown edges through InputState's per-chunk `onset` echo
    (the same protocol as the stream client: seq matched, first frame index of the key's slot)."""

    def __init__(self):
        self.t_key: dict[int, float] = {}
        self.targets: dict[int, dict] = {}   # chunk -> {seq, t, first_idx, t_vis, vis_idx}
        self.onset_ms: list[float] = []
        self.vis_ms: list[float] = []
        self.taps = self.lost = 0
        self.last: float | None = None
        self._last_chunk = -1
        self._vis_pending: tuple[int, float] | None = None

    def keydown(self, seq: int, t: float) -> None:
        self.t_key[seq] = t
        self.taps += 1

    def absorb(self, consumed) -> None:
        """Read InputState.consumed (chunk, summary) entries not seen before."""
        for chunk, s in list(consumed):
            if chunk <= self._last_chunk:
                continue
            self._last_chunk = chunk
            if s.get("reset"):
                continue
            vis = s.get("onset_vis")
            if vis and self._vis_pending and vis["seq"] == self._vis_pending[0]:
                if vis["idx_vis"] is not None:
                    tgt = self.targets.setdefault(chunk, dict(seq=None, t=0.0, first_idx=None, t_vis=0.0, vis_idx=None))
                    tgt["t_vis"], tgt["vis_idx"] = self._vis_pending[1], vis["idx_vis"]
                self._vis_pending = None
            on = s.get("onset")
            if on and on["seq"] in self.t_key:
                t = self.t_key.pop(on["seq"])
                tgt = self.targets.setdefault(chunk, dict(seq=None, t=0.0, first_idx=None, t_vis=0.0, vis_idx=None))
                tgt.update(seq=on["seq"], t=t, first_idx=on["idx"])
                if on["idx_vis"] is not None:
                    tgt["t_vis"], tgt["vis_idx"] = t, on["idx_vis"]
                else:
                    self._vis_pending = (on["seq"], t)

    def presented(self, chunk: int, idx: int, now: float) -> None:
        tgt = self.targets.get(chunk)
        if tgt:
            if tgt["first_idx"] is not None and idx >= tgt["first_idx"]:
                self.last = (now - tgt["t"]) * 1000.0
                self.onset_ms.append(self.last)
                tgt["first_idx"] = None
            if tgt["vis_idx"] is not None and idx >= tgt["vis_idx"]:
                self.vis_ms.append((now - tgt["t_vis"]) * 1000.0)
                tgt["vis_idx"] = None
            if tgt["first_idx"] is None and tgt["vis_idx"] is None:
                del self.targets[chunk]
        for c in [c for c in self.targets if c < chunk - 2]:  # whole chunk never shown
            if self.targets[c]["first_idx"] is not None:
                self.lost += 1
            del self.targets[c]


def _p50(xs):
    return statistics.median(xs) if xs else None


def _fmt_ms(xs, last=None):
    if not xs:
        return "-"
    s = f"p50 {_p50(xs):.0f} ms (n={len(xs)}"
    return s + (f", last {last:.0f})" if last is not None else ")")


def play_loop(src, control: InputState, display, *, fps: int = 16, prefill: int = 6, trough: float = 2.0,
              surplus: str = "rate", seconds: float | None = None, scripted: bool = False, hud=log.info) -> dict:
    """Run until Esc / window close, `seconds` elapsed, or the source ends. Returns the HUD stats."""
    playout: collections.deque = collections.deque()
    rc = RateController(src.frames_per_chunk, target_trough=trough, surplus=surplus)
    rc.nominal_fps = float(fps)
    k2p = KeyToPixel()
    stop = threading.Event()
    done = threading.Event()
    stats = dict(presented=0, underruns=0, warmup_dropped=0, throttled=0)
    hold = int(0.5 * src.frames_per_chunk) + int(trough)

    def throttle():  # --surplus wait: the next chunk starts once the frames queued or on the GPU are down to `hold`
        waited = False
        while not stop.is_set() and started[0] and len(playout) + src.gpu_inflight > hold:
            waited = True
            time.sleep(0.005)
        if waited:
            stats["throttled"] += 1

    started = [False]
    if surplus == "wait":
        src.throttle = throttle

    def consume():
        try:
            for f in src.frames():
                if stop.is_set():
                    break
                if not src.ready.is_set():
                    stats["warmup_dropped"] += 1
                    continue
                playout.append(f)
                rc.arrival(time.monotonic())
        finally:
            done.set()

    threading.Thread(target=consume, name="play-consume", daemon=True).start()
    interval = 1.0 / fps
    t_start = time.monotonic()
    script = ScriptedKeys(t_start) if scripted else None
    seq, prev_held = 0, set()
    next_due = None
    t_hud = t_start
    t_underrun = float("-inf")
    hud_presented = 0
    try:
        while True:
            now = time.monotonic()
            held, reset, quit_, dx, dy = (script or display).poll()
            if script is not None:
                display.poll()  # keep the window's event queue drained (Esc still quits)
            if held and not prev_held:
                seq += 1
                k2p.keydown(seq, now)
            control.update(held, dx, dy, seq=seq, reset=reset)
            prev_held = held
            k2p.absorb(control.consumed)
            if quit_ or (seconds is not None and now - t_start >= seconds) or src.error is not None:
                break
            if done.is_set() and not playout:
                break
            if not started[0]:
                if len(playout) >= prefill or (done.is_set() and playout):
                    started[0] = True
                    next_due = now
                    hud("playout started (prefill %d frames)", prefill)
            elif now >= next_due:
                if playout:
                    f = playout.popleft()
                    display.present(f.rgb)
                    stats["presented"] += 1
                    k2p.presented(f.chunk, f.idx, time.monotonic())
                    next_due += interval * rc.period_scale(now, len(playout))
                else:
                    if now - t_underrun >= interval:  # one hard underrun per missed frame period, not per pass
                        stats["underruns"] += 1
                        t_underrun = now
                    next_due = now  # the next frame is shown the moment it arrives
            if now - t_hud >= HUD_EVERY:
                fps_now = (stats["presented"] - hud_presented) / (now - t_hud)
                hud_presented, t_hud = stats["presented"], now
                rows = src.timing_rows
                spc = rows[-1]["s_per_chunk"] if rows else float("nan")
                warm = "" if src.ready.is_set() else f" | warming up {src.warm_progress}/{src.warm_chunks} chunks"
                line = (f"fps {fps_now:4.1f} | {spc:.2f} s/chunk | key->pixel onset {_fmt_ms(k2p.onset_ms, k2p.last)} "
                        f"visible {_fmt_ms(k2p.vis_ms)} | taps {k2p.taps} lost {k2p.lost} | queue {len(playout)} rate {rc.last_rate:.3f} "
                        f"underruns {stats['underruns']}{warm}")
                hud("%s", line)
                display.set_caption(f"lingbot play  {line}")
            # an empty queue is waited out at the poll rate: a busy spin here starves the generation thread
            # through the GIL (each kernel launch gives the GIL up and waits a switch interval, ~5 ms, to get
            # it back: a chunk's launches then take seconds, and the queue stays empty -- the boundary stall)
            sleep_until = min(next_due, now + 1.0 / POLL_HZ) if started[0] and playout else now + 1.0 / POLL_HZ
            dt = sleep_until - time.monotonic()
            if dt > 0:
                time.sleep(dt)
    finally:
        stop.set()
        src.close()
        display.close()
    rows = src.timing_rows
    steady = [r["s_per_chunk"] for r in rows[1:] if r["s_per_chunk"] == r["s_per_chunk"]]
    stats.update(k2p_onset_ms=k2p.onset_ms, k2p_vis_ms=k2p.vis_ms, taps=k2p.taps, lost=k2p.lost, chunks=len(rows),
                 s_per_chunk_median=_p50(steady), dropped_chunks=src.dropped_chunks, rollouts=src.rollouts,
                 frames_per_chunk=src.frames_per_chunk)
    return stats


def summary_lines(stats: dict, fps: int = 16) -> list[str]:
    spc = stats["s_per_chunk_median"]
    lines = [f"PLAY chunks={stats['chunks']} rollouts={stats['rollouts']} presented={stats['presented']} "
             f"underruns={stats['underruns']} dropped_chunks={stats['dropped_chunks']} throttled={stats['throttled']}"]
    if spc:
        lines.append(f"PLAY s_per_chunk_median={spc:.3f} -> as-played FPS {stats['frames_per_chunk'] / spc:.1f} (real time = {fps})")
    lines.append(f"PLAY key->pixel onset {_fmt_ms(stats['k2p_onset_ms'])} visible {_fmt_ms(stats['k2p_vis_ms'])} "
                 f"taps={stats['taps']} lost={stats['lost']}")
    return lines
