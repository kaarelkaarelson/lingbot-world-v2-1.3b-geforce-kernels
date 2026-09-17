"""Live camera control: held keys / mouse deltas -> per-chunk framewise relative poses.

Frontend-agnostic: the window (lingbot/play/window.py) or a WebSocket server feeds `InputState.update`
with KeyboardEvent.code-style key names (KEY_AXES) and reads the pose provider's per-chunk summary back.

`CameraMotionPlanner` is copied from reactor-cookbook `models/lingbot-world-v2/lingbot_world_v2_camera.py`
(reactor-team, Apache-2.0): it is the mapping reactor.inc uses to drive this checkpoint, so the
pose convention (OpenCV, framewise relative, unit-normalised translation, identity for the first
latent of a rollout) matches what the model was served with. `InputState` is ours: the held
keys and mouse deltas the frontend reports, sampled once per chunk.
"""
from __future__ import annotations

import os
import threading
import time
import numpy as np

FPS = 16
TEMPORAL_STRIDE = 4          # RGB frames per latent
SLOT_SECONDS = TEMPORAL_STRIDE / FPS  # 0.25 s of input per latent
ROTATION_DEG_PER_S = 45.0    # reactor-cookbook lingbot_world_v2.yaml


class RolloutReset(Exception):
    """Raised from the pose provider to end the current rollout early (R key)."""


class CameraMotionPlanner:
    """Build native four-latent camera blocks while preserving camera pose."""

    def __init__(self, fps: float = FPS, rotation_degrees_per_second: float = ROTATION_DEG_PER_S) -> None:
        self._fps = fps
        self._rotation_speed = rotation_degrees_per_second
        self.reset()

    def reset(self, initial_c2w: np.ndarray | None = None) -> None:
        self._current_c2w = np.eye(4, dtype=np.float64) if initial_c2w is None else initial_c2w.astype(np.float64).copy()
        self._first_chunk = True

    def plan_chunk(self, *, forward: float, strafe: float, vertical: float, pitch: float, yaw: float,
                   roll: float, latent_frames: int = 4, temporal_stride: int = TEMPORAL_STRIDE,
                   normalize: bool = True, time_scale: float = 1.0) -> np.ndarray:
        """Return framewise relative OpenCV poses [latent_frames, 4, 4] for one chunk.

        The first latent of a fresh rollout is the anchor and receives identity; every later
        latent receives one integrated transform. Translation is normalised across the chunk
        exactly like upstream ``compute_relative_poses(..., framewise=True)``.
        time_scale: wall seconds a latent is shown for, relative to the nominal stride/fps (1/playback rate);
        the per-latent rotation and translation step scale with it, translation re-bounded to unit norm."""
        controls = (forward, strafe, vertical, pitch, yaw, roll)
        if any(not -1.0 <= v <= 1.0 for v in controls):
            raise ValueError("camera controls must be between -1 and 1")
        translation = _bounded_vector(*(_bounded_vector(strafe, -vertical, forward) * time_scale))
        rotation = _bounded_vector(pitch, yaw, roll)
        seconds = temporal_stride / self._fps * time_scale
        pitch_step, yaw_step, roll_step = np.radians(rotation * self._rotation_speed * seconds)
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = _rotation_z(float(roll_step)) @ _rotation_y(float(yaw_step)) @ _rotation_x(float(-pitch_step))
        delta[:3, 3] = translation

        framewise = np.empty((latent_frames, 4, 4), dtype=np.float64)
        first_index = 0
        if self._first_chunk:
            framewise[0] = np.eye(4, dtype=np.float64)
            first_index = 1
            self._first_chunk = False
        for index in range(first_index, latent_frames):
            previous = self._current_c2w
            self._current_c2w = previous @ delta
            framewise[index] = np.linalg.inv(previous) @ self._current_c2w

        if normalize:
            normalize_translations(framewise)
        return np.ascontiguousarray(framewise, dtype=np.float32)


def normalize_translations(framewise: np.ndarray) -> np.ndarray:
    """Unit-normalise the chunk's translations by their max norm (upstream `compute_relative_poses` convention).
    The cookbook tests `> 0`; after any prior motion a pure-rotation chunk leaves ~1e-16 of
    inverse-product noise, which that test would scale up to a unit translation."""
    translations = framewise[:, :3, 3]
    max_norm = float(np.linalg.norm(translations, axis=-1).max())
    if max_norm > 1e-6:
        framewise[:, :3, 3] = translations / max_norm
    else:
        framewise[:, :3, 3] = 0.0
    return framewise


def _bounded_vector(x: float, y: float, z: float) -> np.ndarray:
    v = np.array([x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1.0 else v


def _rotation_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Keys the frontend reports (KeyboardEvent.code names) -> (axis, sign)
KEY_AXES = {
    "KeyW": ("forward", 1.0), "KeyS": ("forward", -1.0),
    "KeyD": ("strafe", 1.0), "KeyA": ("strafe", -1.0),
    "KeyE": ("vertical", 1.0), "KeyQ": ("vertical", -1.0),
    "ArrowUp": ("pitch", 1.0), "ArrowDown": ("pitch", -1.0),
    "ArrowRight": ("yaw", 1.0), "ArrowLeft": ("yaw", -1.0),
}
MOUSE_PIXELS_PER_FULL_AXIS = 1200.0  # movementX per second of look that means "full-rate yaw" (45 deg/s); ~0.04 deg/px
WALK_SCALE = 0.5                    # without Shift; translation magnitude only shapes the direction mix


def slot_first_frame(slot: int, first_chunk: bool, stride: int = TEMPORAL_STRIDE) -> int:
    """Index within the chunk of the first RGB frame decoded from latent `slot`. A rollout's first chunk has
    the 1+4+4+4 layout (the anchor latent decodes to one frame), every later chunk 4+4+4+4."""
    if first_chunk:
        return 0 if slot == 0 else 1 + stride * (slot - 1)
    return stride * slot


class InputState:
    """Held keys + mouse deltas from the frontend, integrated into per-latent action slots.

    The frontend reports the held set at 30-60 Hz; every report closes a piecewise-constant segment
    of the edge log. `sample()` splits the time since the previous sample into `slots` equal
    bins (4 latents x 250 ms for a 1 s chunk) and returns one axes dict per bin: a key held for
    40 % of a bin gives 0.4 on its axis in that bin only, so taps land in the latent they were
    made in and a key held across a chunk boundary does not become a full-second move.

    `mode="history"` (default) is that replay: history-faithful, one chunk period of lag for every
    key. `mode="hold"` is a zero-order hold (what Reactor and Waypoint-1.5 do): every slot gets the
    keys held *at the sample*, so a held key is responsive from frame 0 of the next chunk instead of
    from the slot it was pressed in (-P/2 key->pixel on average, P = chunk period) at the price of a
    release overshoot of up to one chunk. A key pressed and released inside the window (a tap) would
    then never land, so it is carried over: its measured held time is packed front-aligned from slot 0
    (a 0.4 s tap = slot 0 full + 0.6 of slot 1). Keys that were already held at the previous sample
    were rendered for that whole chunk by the hold, so their release inside the window is not carried
    (no double render; a re-tap of the same key inside the same window is lost). Mouse deltas are
    accumulated per slot in both modes."""

    MODES = ("history", "hold")

    def __init__(self, clock=None, slots: int = 4, mode: str = "history"):
        if mode not in self.MODES:
            raise ValueError(f"input mode must be one of {self.MODES}, got {mode!r}")
        self._lock = threading.Lock()
        self._clock = clock or time.monotonic
        self.slots = slots
        self.mode = mode
        self._keys: set[str] = set()
        self._held_at_sample: frozenset = frozenset()   # hold mode: keys applied by the previous sample
        self._t_last = self._clock()
        self._t_sample = self._t_last
        self._segments: list[tuple[float, float, frozenset]] = []   # (t0, t1, keys held)
        self._mouse: list[tuple[float, float, float]] = []            # (t, dx, dy)
        self._reset = False
        self._onsets: list[tuple[float, int | None]] = []             # (t, seq) idle -> held edges, oldest first
        self._pending_vis: tuple[int | None, frozenset] | None = None  # onset whose >= 0.5 slot is still to come
        self.seq = 0            # last frontend seq applied
        self.playback_rate = 1.0  # wall-time playout speed relative to FPS, written by the transport's rate controller
        self.chunk_offset = 0   # LiveSource adds the rollout offset so acks match the transports' chunk ids
        self.consumed = []      # (chunk_id, summary axes) history for the HUD

    def _close_segment(self):
        now = self._clock()
        if self._keys and now > self._t_last:
            self._segments.append((self._t_last, now, frozenset(self._keys)))
        self._t_last = now

    def update(self, keys, dx: float = 0.0, dy: float = 0.0, seq: int | None = None, reset: bool = False):
        with self._lock:
            self._close_segment()
            if keys and not self._keys:
                self._onsets.append((self._t_last, None if seq is None else int(seq)))
            self._keys = set(keys)
            if dx or dy:
                self._mouse.append((self._t_last, float(dx), float(dy)))
            if seq is not None:
                self.seq = int(seq)
            if reset:
                self._reset = True

    def release_all(self):
        with self._lock:
            self._close_segment()
            self._keys.clear()
            self._mouse.clear()

    def sample(self, chunk_id: int) -> list[dict]:
        """One axes dict per slot for the next chunk; consumes the edge log and the reset flag."""
        with self._lock:
            self._close_segment()
            t0, t1 = self._t_sample, self._t_last
            self._t_sample = t1
            # window = the last slots x 250 ms only: after a stall or a rollout reset the older input is
            # discarded instead of diluting the slots (M3)
            t0 = max(t0, t1 - self.slots * SLOT_SECONDS)
            interval = max(t1 - t0, 1e-3)
            width = interval / self.slots
            held = [dict() for _ in range(self.slots)]
            held_now = frozenset(self._keys)
            carried: dict[str, float] = {}
            if self.mode == "hold":
                for i in range(self.slots):
                    for k in held_now:
                        held[i][k] = 1.0
                # tap carry-over: the released keys' segments, replayed with their relative timing but
                # shifted so the earliest tap edge starts at slot 0 (sequential taps stay sequential)
                taps = []
                for a, b, keys in self._segments:
                    a, b = max(a, t0), min(b, t1)
                    keys = keys - held_now - self._held_at_sample
                    if b <= a or not keys:
                        continue
                    taps.append((a, b, keys))
                    for k in keys:
                        carried[k] = carried.get(k, 0.0) + (b - a)
                shift = min((a for a, _, _ in taps), default=t0) - t0
                for a, b, keys in taps:
                    a, b = a - shift, b - shift
                    for i in range(self.slots):
                        lo, hi = t0 + i * width, t0 + (i + 1) * width
                        ov = min(b, hi) - max(a, lo)
                        if ov > 0:
                            for k in keys:
                                held[i][k] = held[i].get(k, 0.0) + ov / width
                self._held_at_sample = held_now
            else:
                for a, b, keys in self._segments:
                    a, b = max(a, t0), min(b, t1)
                    if b <= a:
                        continue
                    for i in range(self.slots):
                        lo, hi = t0 + i * width, t0 + (i + 1) * width
                        ov = min(b, hi) - max(a, lo)
                        if ov > 0:
                            for k in keys:
                                held[i][k] = held[i].get(k, 0.0) + ov / width
            mouse = [[0.0, 0.0] for _ in range(self.slots)]
            for t, dx, dy in self._mouse:
                if t < t0:  # older than the window (stall/reset): dropped, not folded into slot 0
                    continue
                i = min(self.slots - 1, int((t - t0) / width))
                mouse[i][0] += dx
                mouse[i][1] += dy
            onset = None
            pending_vis, self._pending_vis = self._pending_vis, None
            if pending_vis is not None:            # last window's onset: its >= 0.5 slot is in this window if still held
                seq_on, keys_on = pending_vis
                vis = next((i for i in range(self.slots) if any(held[i].get(k, 0.0) >= 0.5 for k in keys_on)), None)
                pending_vis = {"seq": seq_on, "slot_vis": vis, "idx_vis": None if vis is None else slot_first_frame(vis, chunk_id == 0)}
            onsets_in = [(t, sq) for t, sq in self._onsets if t0 <= t < t1]
            self._onsets = [(t, sq) for t, sq in self._onsets if t >= t1]
            if onsets_in:
                t_on, seq_on = onsets_in[0]
                keys_on = next((ks for a, b, ks in self._segments if a <= t_on < b or a == t_on), frozenset())
                # hold mode: a key that reached the window is applied from slot 0 (held) or carried into slot 0
                slot = 0 if self.mode == "hold" else min(self.slots - 1, int((t_on - t0) / width))
                vis = next((i for i in range(slot, self.slots) if any(held[i].get(k, 0.0) >= 0.5 for k in keys_on)), None)
                if vis is None and self._keys and keys_on & self._keys:
                    self._pending_vis = (seq_on, keys_on)
                onset = {"seq": seq_on, "slot": slot, "slot_vis": vis, "idx": slot_first_frame(slot, chunk_id == 0),
                         "idx_vis": None if vis is None else slot_first_frame(vis, chunk_id == 0)}
            self._segments.clear()
            self._mouse.clear()
            reset, self._reset = self._reset, False
            t_wall = time.time()
            t0_wall_ms = (t_wall - (self._clock() - t0)) * 1000.0  # window start on the wall clock, for the ack
        slots = []
        for i in range(self.slots):
            axes = dict(forward=0.0, strafe=0.0, vertical=0.0, pitch=0.0, yaw=0.0, roll=0.0)
            for k, f in held[i].items():
                if k in KEY_AXES:
                    axis, sign = KEY_AXES[k]
                    axes[axis] += sign * min(1.0, f)
            shift = max(held[i].get("ShiftLeft", 0.0), held[i].get("ShiftRight", 0.0))
            scale = WALK_SCALE + (1.0 - WALK_SCALE) * min(1.0, shift)
            for a in ("forward", "strafe", "vertical"):
                axes[a] = max(-1.0, min(1.0, axes[a])) * scale
            # mouse: a full axis is MOUSE_PIXELS_PER_FULL_AXIS per second of look, so per slot scale by the slot width
            per_slot = MOUSE_PIXELS_PER_FULL_AXIS * width
            axes["yaw"] = max(-1.0, min(1.0, axes["yaw"] + mouse[i][0] / per_slot))
            axes["pitch"] = max(-1.0, min(1.0, axes["pitch"] - mouse[i][1] / per_slot))
            slots.append(axes)
        summary = {k: sum(sl[k] for sl in slots) / len(slots) for k in slots[0]}
        summary["seq"] = self.seq
        summary["reset"] = reset
        summary["first"] = chunk_id == 0     # a rollout's first chunk has the 1+4+4+4 frame layout
        summary["t_sampled_ms"] = t_wall * 1000.0
        summary["t0_ms"] = t0_wall_ms          # ack: the frontend maps its key time to a slot (M2)
        summary["slot_ms"] = width * 1000.0
        summary["onset"] = onset            # the window's first idle -> held edge: the key->pixel sample (seq, slot, slot_vis)
        summary["onset_vis"] = pending_vis  # previous window's onset reaching >= 0.5 weight only now
        summary["input_mode"] = self.mode
        if self.mode == "hold":  # ack/HUD: what the zero-order hold applied to every slot, and the taps carried into slot 0
            summary["held"] = sorted(held_now)
            summary["carried"] = {k: round(v, 3) for k, v in sorted(carried.items())}
        summary["slots"] = slots
        self.consumed.append((chunk_id + self.chunk_offset, summary))
        del self.consumed[:-64]
        return slots, summary


def make_pose_provider(state: InputState, planner: CameraMotionPlanner | None = None, walltime: bool | None = None):
    """pipe.pose_provider for lingbot/play/live.py: samples the input at each chunk start.

    walltime (default LINGBOT_POSE_WALLTIME=1): scale each latent's step by the wall time it is shown for,
    SLOT_SECONDS / state.playback_rate, so 45 deg/s commanded reads 45 deg/s on screen under `--surplus rate`
    (at +15 % playback the 0.25 s step shows as 52 deg/s). The model still sees one bounded per-latent
    delta; only the magnitude a given command maps to changes (11.25 -> 9.8 deg per latent at 1.15)."""
    planner = planner or CameraMotionPlanner()
    if walltime is None:
        walltime = os.environ.get("LINGBOT_POSE_WALLTIME") == "1"

    def provider(chunk_id: int, n_latents: int) -> np.ndarray:
        slots, summary = state.sample(chunk_id)
        time_scale = 1.0 / state.playback_rate if walltime else 1.0
        summary["time_scale"] = time_scale   # in the `applied` ack, so the frontend can see the scaling
        if summary["reset"] and chunk_id > 0:
            planner.reset()
            raise RolloutReset()
        if chunk_id == 0:
            planner.reset()
        # one integrated pose per latent from that latent's slot. Translation magnitude is emitted as-is
        # (run 1.0, walk WALK_SCALE, a 40 % tap 0.4x): the model was trained with per-clip max normalisation,
        # so 1.0 means "the clip's fastest latent" and sub-unit values are ordinary speeds. The cookbook's
        # per-chunk max normalisation would turn every moving latent into 1.0 (walk == run, always at the
        # top of the trained distribution); every other LingBot serving stack keeps a fixed divisor instead.
        if len(slots) != n_latents:  # resample the slots onto the latent count
            slots = [slots[min(len(slots) - 1, int(i * len(slots) / n_latents))] for i in range(n_latents)]
        rel = np.concatenate([
            planner.plan_chunk(forward=a["forward"], strafe=a["strafe"], vertical=a["vertical"], pitch=a["pitch"],
                               yaw=a["yaw"], roll=a["roll"], latent_frames=1, normalize=False, time_scale=time_scale)
            for a in slots])
        return np.ascontiguousarray(rel, dtype=np.float32)
    return provider
