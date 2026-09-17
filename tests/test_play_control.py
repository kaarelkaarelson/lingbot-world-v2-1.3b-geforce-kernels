import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from lingbot.play.control import CameraMotionPlanner, InputState, make_pose_provider, RolloutReset, slot_first_frame


def test_planner_first_latent_identity_then_integrates():
    p = CameraMotionPlanner()
    a = p.plan_chunk(forward=1, strafe=0, vertical=0, pitch=0, yaw=0, roll=0)
    assert a.shape == (4, 4, 4) and a.dtype == np.float32
    assert np.allclose(a[0], np.eye(4))
    # forward = +z in OpenCV, unit-normalised across the chunk
    assert np.allclose(a[1:, :3, 3], [[0, 0, 1]] * 3, atol=1e-6)
    assert np.allclose(a[1:, :3, :3], np.eye(3), atol=1e-6)
    b = p.plan_chunk(forward=0, strafe=0, vertical=0, pitch=0, yaw=1, roll=0)
    assert not np.allclose(b[0], np.eye(4))  # second chunk: no anchor
    # 45 deg/s * 0.25 s per latent = 11.25 deg yaw per latent
    ang = np.degrees(np.arctan2(b[0, 0, 2], b[0, 0, 0]))
    assert abs(ang - 11.25) < 1e-4
    assert np.allclose(b[:, :3, 3], 0)


def test_planner_idle_is_identity():
    p = CameraMotionPlanner()
    p.plan_chunk(forward=0, strafe=0, vertical=0, pitch=0, yaw=0, roll=0)
    b = p.plan_chunk(forward=0, strafe=0, vertical=0, pitch=0, yaw=0, roll=0)
    assert np.allclose(b, np.eye(4)[None].repeat(4, 0))


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def test_input_state_slots():
    clk = Clock()
    st = InputState(clock=clk)
    st.update(["KeyW", "KeyD"], dx=600, dy=-300, seq=7)
    clk.t = 1.0                     # held for the whole 1 s chunk -> every slot
    slots, summary = st.sample(0)
    assert len(slots) == 4 and summary["seq"] == 7 and summary["reset"] is False
    assert all(a["forward"] == pytest.approx(0.5) and a["strafe"] == pytest.approx(0.5) for a in slots)
    assert slots[0]["yaw"] == pytest.approx(1.0) and slots[0]["pitch"] == pytest.approx(1.0)  # mouse at t=0 lands in slot 0 (300 px per 0.25 s slot = full rate)
    assert slots[1]["yaw"] == 0
    clk.t = 1.4; st.update([])      # tap: W+D held for the first 0.4 s of the next chunk only
    clk.t = 2.0
    slots, _ = st.sample(1)
    assert slots[0]["forward"] == pytest.approx(0.5) and slots[1]["forward"] == pytest.approx(0.6 * 0.5)
    assert slots[2]["forward"] == 0 and slots[3]["forward"] == 0
    st.update(["KeyW", "ShiftLeft"], seq=8); clk.t = 3.0
    assert st.sample(2)[0][3]["forward"] == pytest.approx(1.0)
    st.update(["KeyW", "KeyS"]); clk.t = 4.0
    assert st.sample(3)[0][0]["forward"] == pytest.approx(0.0)
    st.update([], dx=10 ** 6); clk.t = 5.0
    assert st.sample(4)[0][0]["yaw"] == 1.0  # clamped; the delta at t=4.0 lands in the first slot
    clk.t = 6.0
    assert all(a["forward"] == 0.0 for a in st.sample(5)[0])  # nothing held: identity


def test_provider_per_slot_poses():
    clk = Clock()
    st = InputState(clock=clk)
    prov = make_pose_provider(st)
    clk.t = 1.0
    prov(0, 4)                      # anchor chunk
    st.update(["KeyW"]); clk.t = 1.5; st.update([]); clk.t = 2.0   # W for slots 0-1 only
    rel = prov(1, 4)
    assert rel.shape == (4, 4, 4)
    # W without Shift = walk: |t| = WALK_SCALE per moving latent (no per-chunk renormalisation)
    assert np.allclose(rel[0, :3, 3], [0, 0, 0.5], atol=1e-6) and np.allclose(rel[1, :3, 3], [0, 0, 0.5], atol=1e-6)
    assert np.allclose(rel[2, :3, 3], 0) and np.allclose(rel[3, :3, 3], 0)
    st.update(["KeyW", "ShiftLeft"]); clk.t = 3.0
    rel = prov(2, 4)
    assert np.allclose(rel[:, :3, 3], [[0, 0, 1.0]] * 4, atol=1e-6)  # run = the clip-max convention


def test_provider_reset_raises_after_first_chunk():
    st = InputState()
    prov = make_pose_provider(st)
    assert prov(0, 4).shape == (4, 4, 4)
    st.update([], reset=True)
    with pytest.raises(RolloutReset):
        prov(1, 4)
    assert prov(0, 4).shape == (4, 4, 4)  # next rollout starts clean


# --- key->pixel onset echo: the server says which window and slot integrated each idle -> held edge ---

def test_onset_slot_and_visible_slot():
    clk = Clock(); st = InputState(clock=clk)
    st.update([], seq=1); clk.t = 1.0; _, s = st.sample(0)
    assert s["onset"] is None and s["onset_vis"] is None
    # 0.4 s tap at 0.3 s into a 1 s window: onset slot 1 (0.25 s slots), 0.2/0.25 = 0.8 of slot 1 -> visible there
    clk.t = 1.3; st.update(["KeyW"], seq=2); clk.t = 1.7; st.update([], seq=3); clk.t = 2.0
    _, s = st.sample(1)
    assert s["onset"] == {"seq": 2, "slot": 1, "slot_vis": 1, "idx": 4, "idx_vis": 4}
    # onset late in slot 1 (0.45): 0.2 of slot 1, all of slot 2 -> onset slot 1, visible slot 2
    clk.t = 2.45; st.update(["KeyW"], seq=4); clk.t = 2.85; st.update([], seq=5); clk.t = 3.0
    _, s = st.sample(2)
    assert s["onset"] == {"seq": 4, "slot": 1, "slot_vis": 2, "idx": 4, "idx_vis": 8}
    # a held key is one onset only: no edge in the next window, the hold continues at full weight
    clk.t = 3.9; st.update(["KeyW"], seq=6); clk.t = 4.0
    _, s = st.sample(3)
    assert s["onset"] == {"seq": 6, "slot": 3, "slot_vis": None, "idx": 12, "idx_vis": None}   # 0.1/0.25 = 0.4 of slot 3: not yet visible
    clk.t = 4.5; st.update(["KeyW"], seq=7); clk.t = 5.0
    _, s = st.sample(4)
    assert s["onset"] is None and s["onset_vis"] == {"seq": 6, "slot_vis": 0, "idx_vis": 0}   # carried over: first >= 0.5 slot of the hold
    # key repeat/tick messages while held do not create onsets; a second key while held is not an edge either
    clk.t = 5.2; st.update(["KeyW", "KeyD"], seq=8); clk.t = 5.6; st.update([], seq=9); clk.t = 6.0
    _, s = st.sample(5)
    assert s["onset"] is None and s["onset_vis"] is None
    # an onset arriving after the sample belongs to the next window, never to this one
    clk.t = 6.01; st.update(["KeyA"], seq=10); clk.t = 7.0
    _, s = st.sample(6)
    assert s["onset"]["seq"] == 10 and s["onset"]["slot"] == 0


def test_onset_released_before_visible_is_echoed_with_no_slot():
    clk = Clock(); st = InputState(clock=clk)
    clk.t = 1.0; st.sample(0)
    clk.t = 1.9; st.update(["KeyW"], seq=2); clk.t = 2.0
    _, s = st.sample(1)
    assert s["onset"]["slot_vis"] is None
    clk.t = 2.05; st.update([], seq=3); clk.t = 3.0      # released 0.05 s into slot 0 of the next window: never reaches 0.5
    _, s = st.sample(2)
    assert s["onset_vis"] == {"seq": 2, "slot_vis": None, "idx_vis": None}


def test_onset_outside_the_window_cap_is_reported_lost():
    # M3: after a > 1 s stall the window is the last 1 s only; an onset before it is not an action the model saw
    clk = Clock(); st = InputState(clock=clk)
    clk.t = 1.0; st.sample(0)
    clk.t = 1.2; st.update(["KeyW"], seq=2); clk.t = 1.6; st.update([], seq=3)
    clk.t = 3.0; _, s = st.sample(1)          # 2 s since the last sample: window = [2.0, 3.0]
    assert s["onset"] is None                  # the client times out seq 2 -> counted as lost, not attributed to slot 0


def test_onset_first_chunk_frame_layout():
    # attribution rule: slot k -> first RGB frame 4k, except a rollout's first chunk (1 + 4 + 4 + 4): 0, 1, 5, 9
    assert [slot_first_frame(k, False) for k in range(4)] == [0, 4, 8, 12]
    assert [slot_first_frame(k, True) for k in range(4)] == [0, 1, 5, 9]
    clk = Clock(); st = InputState(clock=clk)
    clk.t = 0.9; st.update(["KeyW"], seq=1); clk.t = 1.0
    _, s = st.sample(0)                        # chunk 0 of a rollout: onset in slot 3 -> frame 9; 0.4 of the slot, not yet visible
    assert s["first"] and s["onset"] == {"seq": 1, "slot": 3, "slot_vis": None, "idx": 9, "idx_vis": None}
    clk.t = 1.3; st.update([], seq=2); clk.t = 2.0
    _, s = st.sample(1)                        # chunk 1: the hold's >= 0.5 slot is slot 0 -> frame 0
    assert s["onset_vis"] == {"seq": 1, "slot_vis": 0, "idx_vis": 0}
def test_update_bursts_from_edge_and_tick_merge():
    # ?edge=1 sends a message on each key edge and the 33 ms tick keeps running, so two reports of the
    # same held set can land within a ms (or at the same clock instant): no zero-width or duplicated
    # segments, and the sample equals the one a single report per edge would give
    def run(times):
        clk = Clock(); st = InputState(clock=clk)
        clk.t = 1.0; st.sample(0)
        for t, keys in times:
            clk.t = t; st.update(keys)
        assert all(b > a for a, b, _ in st._segments), st._segments
        held = sum(b - a for a, b, _ in st._segments)
        clk.t = 2.0
        return held, st.sample(1)[0]
    plain = run([(1.010, ["KeyW"]), (1.400, [])])
    burst = run([(1.010, ["KeyW"]), (1.010, ["KeyW"]), (1.0105, ["KeyW"]), (1.033, ["KeyW"]), (1.400, []), (1.4004, []), (1.419, [])])
    assert plain[0] == pytest.approx(0.39) and burst[0] == pytest.approx(0.39)
    assert plain[1] == burst[1]
    # the edge timestamp is what lands: W from 1.010 covers 0.96 of slot 0 (1.0-1.25) and 0.6 of slot 1
    assert burst[1][0]["forward"] == pytest.approx(0.96 * 0.5) and burst[1][1]["forward"] == pytest.approx(0.6 * 0.5)
    assert burst[1][2]["forward"] == 0 and burst[1][3]["forward"] == 0


def yaw_deg(rel):
    return np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))


def test_provider_walltime_scales_the_step_by_playback_rate(monkeypatch):
    # rate mode plays a latent for 0.25 / 1.25 s of wall time: with LINGBOT_POSE_WALLTIME the planner integrates
    # that interval (45 deg/s -> 9 deg per latent, walk 0.5 -> 0.4) so on-screen rates match wall time;
    # off (default) it stays at 0.25 s per latent whatever the playout speed
    def chunk(walltime, rate, keys, env=None):
        clk = Clock(); st = InputState(clock=clk); st.playback_rate = rate
        if env is not None:
            monkeypatch.setenv("LINGBOT_POSE_WALLTIME", env)
        prov = make_pose_provider(st, walltime=walltime)
        clk.t = 1.0; prov(0, 4)
        st.update(keys); clk.t = 2.0
        rel = prov(1, 4)
        return rel, st.consumed[-1][1]["time_scale"]
    rel, ts = chunk(True, 1.25, ["ArrowRight", "KeyW"])
    assert ts == pytest.approx(0.8)
    assert np.allclose(yaw_deg(rel), 9.0, atol=1e-4) and np.allclose(rel[:, :3, 3], [[0, 0, 0.4]] * 4, atol=1e-6)
    rel, ts = chunk(False, 1.25, ["ArrowRight", "KeyW"])
    assert ts == 1.0 and np.allclose(yaw_deg(rel), 11.25, atol=1e-4) and np.allclose(rel[:, :3, 3], [[0, 0, 0.5]] * 4, atol=1e-6)
    rel, _ = chunk(None, 1.25, ["ArrowRight"], env="1")          # env switch
    assert np.allclose(yaw_deg(rel), 9.0, atol=1e-4)
    rel, _ = chunk(None, 1.25, ["ArrowRight"], env="0")
    assert np.allclose(yaw_deg(rel), 11.25, atol=1e-4)
    # slower than nominal (0.9 is the controller's floor): the rotation step grows, a run stays bounded at 1.0
    rel, ts = chunk(True, 0.9, ["ArrowRight", "KeyW", "KeyD", "ShiftLeft"])
    assert ts == pytest.approx(1 / 0.9) and np.allclose(yaw_deg(rel), 12.5, atol=1e-4)
    assert np.allclose(np.linalg.norm(rel[:, :3, 3], axis=-1), 1.0, atol=1e-6)
def test_hold_mode_held_key_fills_every_slot():
    clk = Clock()
    st = InputState(clock=clk, mode="hold")
    clk.t = 0.9; st.update(["KeyW"])      # pressed 0.1 s before the sample: history would give slot 3 = 0.4
    clk.t = 1.0
    slots, summary = st.sample(0)
    assert all(a["forward"] == pytest.approx(0.5) for a in slots)   # zero-order hold: responsive from frame 0
    assert summary["input_mode"] == "hold" and summary["held"] == ["KeyW"] and summary["carried"] == {}
    clk.t = 2.0                           # still held: full slots again, nothing carried
    slots, summary = st.sample(1)
    assert all(a["forward"] == pytest.approx(0.5) for a in slots) and summary["carried"] == {}


def test_hold_mode_tap_carried_into_slot_0():
    clk = Clock()
    st = InputState(clock=clk, mode="hold")
    clk.t = 0.2; st.update(["KeyW"]); clk.t = 0.6; st.update([])   # 0.4 s tap, released before the sample
    clk.t = 1.0
    slots, summary = st.sample(0)
    # front-aligned: 0.4 s = 1.6 slot widths -> slot 0 full, 0.6 of slot 1, nothing later (history: slots 0-2 = 0.2/1/0.4)
    assert [a["forward"] for a in slots] == pytest.approx([0.5, 0.6 * 0.5, 0.0, 0.0])
    assert summary["held"] == [] and summary["carried"] == {"KeyW": 0.4}
    clk.t = 1.3; st.update(["KeyD"]); clk.t = 1.4; st.update([])   # 0.1 s tap -> 0.4 of slot 0 only
    clk.t = 2.0
    slots, summary = st.sample(1)
    assert [a["strafe"] for a in slots] == pytest.approx([0.4 * 0.5, 0.0, 0.0, 0.0])
    assert all(a["forward"] == 0.0 for a in slots)   # the earlier tap was consumed
    clk.t = 3.0
    assert all(a["strafe"] == 0.0 for a in st.sample(2)[0])


def test_hold_mode_release_leaves_next_chunk_empty():
    clk = Clock()
    st = InputState(clock=clk, mode="hold")
    st.update(["KeyW"]); clk.t = 1.0
    assert all(a["forward"] == pytest.approx(0.5) for a in st.sample(0)[0])
    clk.t = 1.6; st.update([])            # released 0.6 s into the window: already rendered by chunk 0's hold
    clk.t = 2.0
    slots, summary = st.sample(1)
    assert all(a["forward"] == 0.0 for a in slots) and summary["held"] == [] and summary["carried"] == {}


def test_hold_mode_mouse_unchanged():
    for mode in ("history", "hold"):
        clk = Clock()
        st = InputState(clock=clk, mode=mode)
        st.update([], dx=600, dy=-300)
        clk.t = 0.6; st.update([], dx=150)
        clk.t = 1.0
        slots, _ = st.sample(0)
        assert slots[0]["yaw"] == pytest.approx(1.0) and slots[0]["pitch"] == pytest.approx(1.0)
        assert slots[2]["yaw"] == pytest.approx(0.5) and slots[1]["yaw"] == 0 and slots[3]["yaw"] == 0
        clk.t = 2.0
        assert all(a["yaw"] == 0.0 for a in st.sample(1)[0])   # deltas are consumed, not held


def test_input_mode_validated():
    with pytest.raises(ValueError):
        InputState(mode="lead")


def test_hold_mode_sequential_taps_stay_sequential():
    """rt-hold-predict: W then S inside one window must not cancel (front-aligning each key made them concurrent)."""
    clk = Clock()
    st = InputState(clock=clk, slots=4, mode="hold")
    clk.t = 0.05; st.update(["KeyW"])
    clk.t = 0.25; st.update([])
    clk.t = 0.40; st.update(["KeyS"])
    clk.t = 0.60; st.update([])
    clk.t = 0.87
    slots, _ = st.sample(1)
    fwd = [round(s["forward"], 2) for s in slots]
    assert fwd[0] > 0 and fwd[2] < 0, fwd          # W first, S later, both rendered
    assert not all(v == 0 for v in fwd), fwd
