import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from lingbot.play.live import LiveSource, DryPipe, to_uint8_hwc


def test_to_uint8_matches_save_video_rounding():
    fr = torch.tensor([-1.0, 0.0, 1.0, 0.5]).view(1, 1, 1, 4).expand(3, 2, 1, 4).clone()
    u8 = to_uint8_hwc(fr)
    assert u8.shape == (2, 1, 4, 3) and u8.dtype == torch.uint8
    assert u8[0, 0, :, 0].tolist() == [0, 128, 255, 191]


def test_dry_pipe_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGBOT_VAE_FUSED", "1")
    monkeypatch.setenv("LINGBOT_VAE_STREAM", "1")
    pipe = DryPipe(n_chunks=4, chunk_seconds=0.05, h=32, w=64)
    src = LiveSource(pipe, img=None, action_path="/dev/null", prompt="p",
                     timing_tsv=str(tmp_path / "t.tsv"), width=64, height=32)
    frames = list(src.frames())
    assert src.error is None
    per_chunk = {}
    for f in frames:
        per_chunk.setdefault(f.chunk, []).append(f)
    assert sorted(per_chunk) == [0, 1, 2, 3]
    assert len(per_chunk[0]) == 13 and all(len(per_chunk[c]) == 16 for c in (1, 2, 3))
    assert all(f.rgb.shape == (32, 64, 3) and f.rgb.dtype == np.uint8 for f in frames)
    assert [f.idx for f in per_chunk[1]] == list(range(16))
    # frames own their memory (pinned buffers are recycled)
    assert all(f.rgb.flags.owndata for f in frames)
    # chunk N is ready after chunk N+1 started (decode overlaps the next chunk's DiT)
    assert per_chunk[0][0].t_ready >= per_chunk[1][0].chunk_gen_start
    rows = src.timing_rows
    assert [r["chunk"] for r in rows] == [0, 1, 2, 3] and rows[0]["s_per_chunk"] != rows[0]["s_per_chunk"]  # nan
    assert all(0.04 < r["s_per_chunk"] < 0.5 for r in rows[1:])
    txt = (tmp_path / "t.tsv").read_text().splitlines()
    assert txt[0].split("\t") == ["chunk", "gen_start", "dit_end", "decode_end", "s_per_chunk", "frames"] and len(txt) == 5
    assert src.dropped_chunks == 0
    src.close()


def test_requires_env(monkeypatch):
    monkeypatch.delenv("LINGBOT_VAE_STREAM", raising=False)
    monkeypatch.setenv("LINGBOT_VAE_FUSED", "1")
    with pytest.raises(RuntimeError):
        LiveSource(DryPipe(), None, "/dev/null", "p")


def test_sink_drops_when_pool_exhausted(monkeypatch):
    monkeypatch.setenv("LINGBOT_VAE_FUSED", "1")
    monkeypatch.setenv("LINGBOT_VAE_STREAM", "1")
    pipe = DryPipe(n_chunks=1, h=8, w=8)
    src = LiveSource(pipe, None, "/dev/null", "p", pool_chunks=1, width=8, height=8)
    list(src.frames())
    # drain the pooled buffers (pool_chunks * chunk_size of them), then call the sink directly
    bufs = []
    while True:
        b = src.pool.get()
        if b is None:
            break
        bufs.append(b)
    assert bufs
    buf = bufs[0]
    src._sink(9, torch.zeros(3, 4, 8, 8), None, time.monotonic())
    assert src.dropped_at_sink == 1 and src.dropped_chunks == 1
    for b in bufs:
        src.pool.put(b)


def test_loop_continues_across_rollouts(monkeypatch):
    monkeypatch.setenv("LINGBOT_VAE_FUSED", "1")
    monkeypatch.setenv("LINGBOT_VAE_STREAM", "1")
    pipe = DryPipe(n_chunks=3, chunk_seconds=0.02, h=8, w=8)
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8, loop=True)
    seen = []
    for f in src.frames():
        if f.idx == 0:
            seen.append(f.chunk)
        if f.chunk >= 7:  # third rollout has started
            break
    src.close()
    assert seen[:8] == [0, 1, 2, 3, 4, 5, 6, 7]  # chunk ids keep counting across rollouts
    assert src.rollouts >= 2 and src.error is None


def test_control_drives_provider_and_reset(monkeypatch):
    from lingbot.play.control import InputState
    monkeypatch.setenv("LINGBOT_VAE_FUSED", "1")
    monkeypatch.setenv("LINGBOT_VAE_STREAM", "1")
    pipe = DryPipe(n_chunks=4, chunk_seconds=0.02, h=8, w=8)
    st = InputState()
    st.update(["KeyW"])
    time.sleep(0.3)  # hold W for one slot before chunk 0 samples (the window is the last 4 x 250 ms)
    time.sleep(0.05)  # W must be held for a measurable share of chunk 0's input window (edge-log integrator)
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8, loop=True, control=st)
    chunks = []
    for f in src.frames():
        if f.idx == 0:
            chunks.append(f.chunk)
        if f.chunk == 1:
            st.update([], reset=True)   # reset mid-rollout -> rollout 2 starts at chunk 4 after chunk 2
        if f.chunk >= 6:
            break
    src.close()
    assert src.rollouts >= 1 and src.error is None
    assert len(pipe.poses) >= 5 and pipe.poses[0].shape == (4, 4, 4)
    assert np.allclose(pipe.poses[0][1, :3, 3], [0, 0, 0.5], atol=1e-6)  # W held (walk) -> forward at WALK_SCALE
    assert chunks == sorted(chunks)


def test_decode_first_per_latent_emission(monkeypatch):
    monkeypatch.setenv("LINGBOT_VAE_FUSED", "1")
    monkeypatch.setenv("LINGBOT_VAE_STREAM", "1")
    pipe = DryPipe(n_chunks=3, chunk_seconds=0.02, h=8, w=8, decode_first=True)
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8)
    frames = list(src.frames())
    assert src.error is None
    ids = [f.frame_id for f in frames]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    per_chunk = {}
    for f in frames:
        per_chunk.setdefault(f.chunk, []).append(f.idx)
    assert per_chunk[0] == list(range(13)) and per_chunk[1] == list(range(16)) and per_chunk[2] == list(range(16))
    assert [r["chunk"] for r in src.timing_rows] == [0, 1, 2]  # one timing row per chunk
    src.close()


@pytest.mark.parametrize("cs,first", [(3, 9), (2, 5)])
def test_chunk_size_plumbing_and_per_latent_offsets(monkeypatch, cs, first):
    monkeypatch.setenv("LINGBOT_VAE_FUSED", "1")
    monkeypatch.setenv("LINGBOT_VAE_STREAM", "1")
    pipe = DryPipe(n_chunks=3, chunk_seconds=0.02, h=8, w=8, decode_first=True)
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8, chunk_size=cs)
    assert src.frames_per_chunk == 4 * cs and src.gen_kwargs["chunk_size"] == cs
    frames = list(src.frames())
    assert src.error is None
    per_chunk = {}
    for f in frames:
        per_chunk.setdefault(f.chunk, []).append(f.idx)
    assert per_chunk[0] == list(range(first)) and per_chunk[1] == list(range(4 * cs)) and per_chunk[2] == list(range(4 * cs))
    ids = [f.frame_id for f in frames]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) and ids[first] == 4 * cs
    src.close()


def test_control_slots_follow_chunk_size(monkeypatch):
    from lingbot.play.control import InputState
    monkeypatch.setenv("LINGBOT_VAE_FUSED", "1")
    monkeypatch.setenv("LINGBOT_VAE_STREAM", "1")
    pipe = DryPipe(n_chunks=2, chunk_seconds=0.02, h=8, w=8)
    st = InputState()
    st.update(["KeyW"])
    time.sleep(0.05)
    src = LiveSource(pipe, None, "/dev/null", "p", width=8, height=8, chunk_size=3, control=st)
    list(src.frames())
    src.close()
    assert st.slots == 3
    assert all(p.shape == (3, 4, 4) for p in pipe.poses)
