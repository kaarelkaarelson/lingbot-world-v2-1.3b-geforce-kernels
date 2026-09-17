"""Playout-rate control for the window: the stream repo's RateController (stream/ws/server.py), without
the codec-bound 'drop' policy (there is no keyframe to cut at locally)."""
from __future__ import annotations

import collections


class RateController:
    """Playout-rate control for a queue fed in bursts (16 frames once per ~1 s, or 4 x 4 frames over the
    last ~0.34 s of each chunk with per-latent emission) and drained one frame per period.

    The queue depth is a sawtooth, so a controller on the *mean* depth lets the trough hit zero: with a
    mean of 6 and a 16-frame burst the depth swings 14 -> -2, which is the ~6 underruns / 15 s measured on
    the pod. This controller works on the trough (the minimum depth over the last `window` s):

    - outer loop, slow: period *= 1 / (1 + clamp((trough_ema - target_trough) / (4 * fpc), -0.05, +0.05));
      settles the trough at `target_trough` frames, tracking a source that runs up to +-5 % off nominal
      (a 0.5 % deficit costs 0.5 % playback speed, invisible).
    - inner loop, immediate: below the low-water mark (depth <= `low`, default 2) the period is stretched
      linearly up to +15 % at depth 1 (a few ms per frame rather than a 62 ms hold); above the high-water
      mark (depth >= target_trough + fpc + 4) it is shrunk down to -10 % to drain a stall backlog.
    - depth 0 is still a hard underrun (nothing to present); it is counted separately from stretched frames.
    - surplus (source faster than real time, 16.2 FPS generated vs 16 played): `surplus="rate"` (default)
      lets the outer loop run up to +25 % fast (the video plays fast, nothing dropped; the cap must sit
      above the source's surplus or the trough loop has no headroom and the queue parks high);
      `surplus="wait"` keeps playback nominal and the presenter throttles the source instead (LiveSource.throttle).
    """

    def __init__(self, fpc: int, target_trough: float = 2.0, low: int = 2, window: float = 1.25,
                 outer_gain_frames: float | None = None, surplus: str = "rate"):
        self.fpc = fpc
        self.target_trough = float(target_trough)
        self.low = low
        self.window = window
        self.high = self.target_trough + fpc + 4
        self.outer_gain = outer_gain_frames or (4.0 * fpc)
        self.surplus = surplus
        # rate: play up to +25 % fast (a generated world has no real time); wait: the source is throttled to the
        # playout, so any speed-up would feed back into the arrival estimate (a +5 % loop, measured) -> nominal only
        self.max_fast = {"rate": 0.25, "wait": 0.0}[surplus]
        self.trough_ema: float | None = None
        self._hist: collections.deque = collections.deque()  # (t, depth)
        # feed-forward: the measured arrival rate (frames/s over the last ~3 s) relative to nominal fps, so a
        # source running fast or slow is matched directly and the P-term only corrects the trough
        # (a P-term alone needs a 6-frame error to reach +13 % — measured 0.4 s of extra queue)
        self.nominal_fps: float | None = None
        self._arrivals: collections.deque = collections.deque()  # arrival timestamps
        self.base_rate = 1.0
        self.stretched = 0
        self.shrunk = 0
        self.last_rate = 1.0

    def observe(self, t: float, depth: int) -> None:
        self._hist.append((t, depth))
        while self._hist and self._hist[0][0] < t - self.window:
            self._hist.popleft()

    @property
    def trough(self) -> int:
        return min(d for _, d in self._hist) if self._hist else 0

    def arrival(self, t: float) -> None:
        """Record one frame entering the playout queue (feed-forward rate estimate)."""
        # a stall or rollout gap is not a rate: with it inside the window the estimate would sit low for the
        # next ~23 chunks and the queue triple (sim: +0.6-1.3 s for 30 s); start fresh, keep the last estimate
        if self._arrivals and self.nominal_fps and t - self._arrivals[-1] > 2.0 * self.fpc / self.nominal_fps:
            self._arrivals.clear()
        self._arrivals.append(t)
        while len(self._arrivals) > 24 * self.fpc:
            self._arrivals.popleft()
        # arrivals are bursty (a chunk at a time), so measure the time to go back exactly k chunks' worth
        # of frames: same burst phase at both ends, unbiased; needs >= 8 chunks of history
        if self.surplus == "wait":
            return  # arrivals follow the playout: no feed-forward
        if self.nominal_fps and len(self._arrivals) > 8 * self.fpc:
            k = (len(self._arrivals) - 1) // self.fpc
            t0 = self._arrivals[-1 - k * self.fpc]
            if t - t0 > 0:
                rate = (k * self.fpc) / (t - t0) / self.nominal_fps
                self.base_rate = max(0.9, min(1.0 + self.max_fast, rate))

    def period_scale(self, t: float, depth: int) -> float:
        """Multiplier for the nominal frame period after presenting a frame that left `depth` frames queued."""
        self.observe(t, depth)
        tr = self.trough
        self.trough_ema = tr if self.trough_ema is None else self.trough_ema + 0.05 * (tr - self.trough_ema)
        outer = self.base_rate * (1.0 + max(-0.05, min(0.05, (self.trough_ema - self.target_trough) / self.outer_gain)))
        outer = max(0.9, min(1.0 + self.max_fast, outer))
        scale = 1.0 / outer
        if depth <= self.low:
            scale *= 1.0 + 0.15 * (self.low + 1 - depth) / (self.low + 1)  # depth 1: +10 %, depth 2: +5 %
            self.stretched += 1
        elif depth >= self.high:
            scale *= max(0.9, 1.0 - 0.10 * (depth - self.high + 1) / self.fpc)
            self.shrunk += 1
        scale = max(scale, 1.0 / (1.0 + self.max_fast))  # the window follows at most this fast
        self.last_rate = 1.0 / scale
        return scale
