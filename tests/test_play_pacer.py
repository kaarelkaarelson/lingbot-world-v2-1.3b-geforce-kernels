"""Fake-clock model of the window's pacer (lingbot/play/pacer.py): frames arrive per the source's burst
pattern, one leaves per paced period. Ported from the stream repo's test_pacer.py without the 'drop' policy."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from lingbot.play.pacer import RateController

FPC, FPS = 16, 16


def arrivals(pattern: str, chunk_s: float, seconds: float, dit_s: float = 0.65, gap: tuple | None = None):
    """(time, n_frames) events. whole: 16 frames at the end of each chunk; per_latent: 4 x 4 frames over the
    last 0.34 s of each chunk (chunk 0: 1 + 4 + 4 + 4). gap=(chunk, seconds): a source stall before that chunk."""
    ev, t, chunk = [], 0.0, 0
    while t < seconds:
        if gap and chunk == gap[0]:
            t += gap[1]
        if pattern == "whole":
            ev.append((t + chunk_s, 13 if chunk == 0 else FPC))
        else:
            groups = [1, 4, 4, 4] if chunk == 0 else [4] * 4
            span = chunk_s - dit_s
            for li, g in enumerate(groups):
                ev.append((t + dit_s + span * (li + 1) / 4, g))
        t += chunk_s
        chunk += 1
    return ev


def simulate(pattern, chunk_s, seconds=120.0, prefill=6, trough=2.0, surplus="rate", gap=None):
    ev = arrivals(pattern, chunk_s, seconds + 5, gap=gap)
    rc = RateController(FPC, target_trough=trough, surplus=surplus)
    rc.nominal_fps = float(FPS)
    interval = 1.0 / FPS
    q: list = []
    produced = 0
    i, t, started, next_due = 0, 0.0, False, 0.0
    hard, hard_late, sent, periods, depths = 0, 0, 0, [], []
    while t < seconds:
        while i < len(ev) and ev[i][0] <= t:
            q.extend(range(produced, produced + ev[i][1])); produced += ev[i][1]
            for _ in range(ev[i][1]):
                rc.arrival(ev[i][0])
            i += 1
        if not started:
            if len(q) < prefill:
                t = ev[i][0] if i < len(ev) else t + 0.01
                continue
            started, next_due = True, t
        if t < next_due:
            t = min(next_due, ev[i][0] if i < len(ev) else next_due)
            continue
        if not q:
            hard += 1
            if t >= 5.0:
                hard_late += 1
            t = ev[i][0] if i < len(ev) else t + 0.01
            next_due = t
            continue
        q.pop(0); sent += 1
        depth = len(q)
        if t >= 20.0:
            depths.append(depth)
        scale = rc.period_scale(t, depth)
        periods.append(interval * scale)
        next_due += interval * scale
    return dict(hard=hard, hard_after_5s=hard_late, sent=sent, stretched=rc.stretched,
                mean_period=sum(periods[80:]) / len(periods[80:]),
                depth_mean=sum(depths) / len(depths), depth_max=max(depths), depth_min=min(depths))


@pytest.mark.parametrize("pattern", ["whole", "per_latent"])
@pytest.mark.parametrize("chunk_s", [0.976, 1.005])
def test_no_hard_underruns_and_tracks_source(pattern, chunk_s):
    r = simulate(pattern, chunk_s)
    assert r["hard_after_5s"] == 0 and r["hard"] <= 2, r
    src_period = chunk_s / FPC
    assert abs(r["mean_period"] - src_period) / src_period < 0.01, r


def test_stretch_is_bounded():
    rc = RateController(FPC, target_trough=2.0)
    scales = [rc.period_scale(t * 0.0625, d) for t, d in enumerate([1, 2, 3, 30, 40])]
    assert 1.0 < scales[0] <= 1.15 * 1.05 and 1.0 < scales[1] <= 1.15 * 1.05
    assert scales[3] < 1.0 and scales[3] >= 0.9 / 1.05


def test_surplus_rate_plays_the_source_speed_without_parking_high():
    # 16.2 FPS generated (0.98 s per 16-frame chunk) and a 3-step 0.87 s source: the queue never parks
    # above the high-water mark and the mean period tracks the source
    for cs in (0.98, 0.87):
        r = simulate("per_latent", cs, surplus="rate")
        assert r["hard_after_5s"] == 0, r
        assert r["depth_mean"] < 2 + FPC + 4 + 2, r
        assert abs(r["mean_period"] - cs / FPC) / (cs / FPC) < 0.01, r


def test_gap_does_not_inflate_queue():
    base = simulate("per_latent", 0.87, surplus="rate")
    gapped = simulate("per_latent", 0.87, surplus="rate", gap=(40, 2.0))
    assert gapped["hard_after_5s"] <= 1, gapped
    assert gapped["depth_max"] <= base["depth_max"] + 2, (base, gapped)
    assert abs(gapped["depth_mean"] - base["depth_mean"]) < 1.0, (base, gapped)


def test_wait_policy_never_plays_fast():
    rc = RateController(FPC, surplus="wait")
    rc.nominal_fps = float(FPS)
    for k in range(400):
        rc.arrival(k * 0.05)   # 20 fps arrivals: a rate the wait policy must not follow
    assert rc.base_rate == 1.0
    assert min(rc.period_scale(20 + t * 0.0625, 30) for t in range(40)) >= 1.0
