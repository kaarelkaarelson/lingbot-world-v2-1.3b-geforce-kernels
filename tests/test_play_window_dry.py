"""The window loop on CPU: DryPipe frames through LiveSource, pacer, key->pixel attribution and a
headless display (SDL_VIDEODRIVER=dummy when pygame is installed, NullDisplay otherwise)."""
import os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pytest

torch = pytest.importorskip("torch")
from lingbot.play import window
from lingbot.play.control import InputState
from lingbot.play.live import DryPipe, LiveSource


def _env(monkeypatch):
    for k, v in {"LINGBOT_VAE_FUSED": "1", "LINGBOT_VAE_STREAM": "1", "LINGBOT_DECODE_FIRST": "1",
                 "LINGBOT_WARM_CHUNKS": "0", "SDL_VIDEODRIVER": "dummy"}.items():
        monkeypatch.setenv(k, v)


def test_loop_presents_frames_and_consumes_a_key_edge(monkeypatch):
    _env(monkeypatch)
    st = InputState()
    pipe = DryPipe(n_chunks=3, chunk_seconds=0.3, h=32, w=64, decode_first=True)
    src = LiveSource(pipe, None, "/dev/null", "p", width=64, height=32, control=st)
    disp = window.open_display(64, 32, "t", headless=True)
    hud = []
    stats = window.play_loop(src, st, disp, scripted=True, hud=lambda fmt, *a: hud.append(fmt % a))
    assert src.error is None
    assert stats["presented"] == 13 + 16 + 16 and stats["chunks"] == 3
    # the scripted W tap (0.1-0.5 s) was integrated by a chunk sample and closed by a presented frame
    assert len(pipe.poses) == 3 and any(abs(p[:, 2, 3]).max() > 0 for p in pipe.poses)
    assert stats["taps"] >= 1 and len(stats["k2p_onset_ms"]) == 1 and stats["lost"] == 0
    assert 300 < stats["k2p_onset_ms"][0] < 3000
    assert stats["underruns"] == 0 and any("key->pixel onset" in h for h in hud)
    assert "key->pixel onset p50" in window.summary_lines(stats)[-1]


def test_key_to_pixel_attribution_matches_the_onset_echo():
    k = window.KeyToPixel()
    k.keydown(1, 10.0)
    # the sample for chunk 5 says: seq 1 landed in slot 1 (frame 4), >= 0.5 weight reached at slot 2 (frame 8)
    k.absorb([(5, dict(reset=False, onset=dict(seq=1, slot=1, idx=4, slot_vis=2, idx_vis=8), onset_vis=None))])
    k.presented(5, 3, 11.0)
    assert k.onset_ms == [] and k.vis_ms == []
    k.presented(5, 4, 11.2)
    assert k.onset_ms == [pytest.approx(1200.0)] and k.vis_ms == []
    k.presented(5, 9, 11.5)   # frame 8 never presented: the >= idx rule keeps the sample
    assert k.vis_ms == [pytest.approx(1500.0)] and 5 not in k.targets
    # an onset whose visible slot only comes in the next window (onset_vis), and a target never shown (lost)
    k.keydown(2, 20.0)
    k.absorb([(6, dict(reset=False, onset=dict(seq=2, slot=3, idx=12, slot_vis=None, idx_vis=None), onset_vis=None))])
    k.absorb([(7, dict(reset=False, onset=None, onset_vis=dict(seq=2, slot_vis=0, idx_vis=0)))])
    assert k.targets[6]["first_idx"] == 12 and k.targets[7]["vis_idx"] == 0
    k.presented(10, 0, 21.0)   # two whole chunks later: 6 and 7 were never shown
    assert k.lost == 1 and 6 not in k.targets and 7 not in k.targets


def test_wait_surplus_throttles_the_source(monkeypatch):
    _env(monkeypatch)
    st = InputState()
    pipe = DryPipe(n_chunks=4, chunk_seconds=0.05, h=8, w=8, decode_first=True)  # far faster than 16 fps
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8, control=st)
    disp = window.open_display(8, 8, "t", headless=True)
    stats = window.play_loop(src, st, disp, surplus="wait", hud=lambda *a: None)
    assert src.error is None and stats["presented"] == 13 + 48
    assert stats["throttled"] >= 1 and stats["underruns"] == 0


def test_cli_smoke_and_play_dry():
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    for args in (["--help"], ["play", "--help"]):
        out = subprocess.run([sys.executable, "-m", "lingbot.cli", *args], cwd=ROOT, env=env, capture_output=True, text=True)
        assert out.returncode == 0 and "play" in out.stdout, out
    out = subprocess.run([sys.executable, "-m", "lingbot.cli", "play", "wall", "--dry"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PLAY chunks=3" in out.stdout and "key->pixel onset p50" in out.stdout, out.stdout
    out = subprocess.run([sys.executable, "-m", "lingbot.cli", "play", "moon", "--dry"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 2 and "invalid choice" in out.stderr


def test_every_scene_is_complete():
    from lingbot.cli import SCENES
    for name, d in SCENES.items():
        for f in ("image.jpg", "prompt.txt", "intrinsics.npy"):
            assert os.path.exists(os.path.join(ROOT, d, f)), f"{name}: {d}/{f}"
        assert len(open(os.path.join(ROOT, d, "prompt.txt")).read().strip()) > 40, name


def test_rollout_boundary_does_not_stall_the_presenter(monkeypatch):
    """The pod stall: at a rollout boundary the playout queue empties (0.5-0.9 s of setup before chunk 0), and a
    presenter that busy-spins on the empty queue starves the generation thread through the GIL (every kernel
    launch gives the GIL up and waits ~5 ms to get it back), so the queue stays empty for tens of seconds."""
    _env(monkeypatch)
    st = InputState()
    # 16 frames per 0.4 s = real time at 40 fps; a 0.5 s gap between rollouts (the pipeline's 0.5-0.9 s), longer
    # than the buffered chunk, so the queue is empty while chunk 0's 300 GIL round trips run
    period, gap = 0.4, 0.5
    pipe = DryPipe(n_chunks=3, chunk_seconds=period, h=8, w=8, decode_first=True, boundary_seconds=gap, ops_per_chunk=300)
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8, control=st, loop=True)
    disp = window.open_display(8, 8, "t", headless=True)
    shown = []
    present = disp.present
    disp.present = lambda rgb: (shown.append(time.monotonic()), present(rgb))
    stats = window.play_loop(src, st, disp, fps=40, seconds=4.5, scripted=True, hud=lambda *a: None)
    gaps = [b - a for a, b in zip(shown, shown[1:])]
    # rollout 2's chunk 0 never arrived within the run under the spin (300 x ~5 ms per launch, then the same
    # again for every chunk that starts on an empty queue); paced, a boundary costs the gap plus chunk 0
    assert src.error is None and src.rollouts >= 2, (stats, max(gaps))
    assert max(gaps) < gap + 1.5 * period, f"presentation stalled for {max(gaps):.2f} s"
    # hard underruns are counted per missed frame period (~20 per boundary at 40 fps), not per loop pass
    assert stats["underruns"] < 80, stats["underruns"]


def test_esc_stops_generation_and_joins_the_threads(monkeypatch, tmp_path):
    _env(monkeypatch)
    st = InputState()
    pipe = DryPipe(n_chunks=4, chunk_seconds=0.1, h=8, w=8, decode_first=True)
    tsv = tmp_path / "t.tsv"
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8, control=st, loop=True, timing_tsv=str(tsv))
    disp = window.open_display(8, 8, "t", headless=True)
    presented = [0]
    present = disp.present
    disp.present = lambda rgb: (presented.__setitem__(0, presented[0] + 1), present(rgb))
    poll = disp.poll

    def poll_with_esc():  # Esc after 20 frames, mid-rollout (posted to pygame's event queue when it is the backend)
        esc = presented[0] >= 20
        if esc and hasattr(disp, "pg"):
            disp.pg.event.post(disp.pg.event.Event(disp.pg.KEYDOWN, key=disp.pg.K_ESCAPE))
        held, reset, quit_, dx, dy = poll()
        return held, reset, quit_ or (esc and not hasattr(disp, "pg")), dx, dy
    disp.poll = poll_with_esc
    t0 = time.monotonic()
    stats = window.play_loop(src, st, disp, hud=lambda *a: None)
    assert time.monotonic() - t0 < 5.0 and 20 <= stats["presented"] < 40
    # close() unwound generate() at the chunk gate and joined both threads; the timing file closed after the host
    assert not src._gen.is_alive() and not src._host.is_alive() and src.error is None
    assert src._timing_f.closed and len(tsv.read_text().splitlines()) == len(src.timing_rows) + 1


def test_headless_seconds_and_taps_count_from_ready(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("LINGBOT_WARM_CHUNKS", "2")   # ready at chunk 2: two chunks of warm-up first
    st = InputState()
    pipe = DryPipe(n_chunks=3, chunk_seconds=0.3, h=8, w=8, decode_first=True)
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8, control=st, loop=True)
    disp = window.open_display(8, 8, "t", headless=True)
    t0 = time.monotonic()
    stats = window.play_loop(src, st, disp, seconds=1.0, scripted=True, hud=lambda *a: None)
    dt = time.monotonic() - t0
    assert 0.5 < stats["warmup_s"] < 1.5 and stats["warmup_dropped"] > 0
    assert dt >= stats["warmup_s"] + 1.0 and 0.9 < stats["played_s"] < 1.6   # the 1.0 s ran after the warm-up
    # no tap was registered before ready, so none could land on a discarded warm-up chunk
    assert stats["lost"] == 0 and stats["taps"] >= 1
    assert "warmup=" in window.summary_lines(stats)[0]
