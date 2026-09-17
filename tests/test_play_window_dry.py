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
    assert stats["underruns"] == 0 and any("key->pixel onset p50" in h for h in hud)


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
    out = subprocess.run([sys.executable, "-m", "lingbot.cli", "play", "--dry"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PLAY chunks=3" in out.stdout and "key->pixel onset p50" in out.stdout, out.stdout
