"""Frame and ChunkQueue: the hand-off from the live model (lingbot/play/live.py) to a presenter
(the window, or a streaming transport).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass
class Frame:
    chunk: int              # chunk index, 0-based
    idx: int                # frame index within the chunk (0..frames_in_chunk-1)
    t_ready: float          # time.monotonic() when `rgb` became available on the host
    chunk_gen_start: float  # time.monotonic() when this chunk's generation started
    rgb: np.ndarray         # (H, W, 3) uint8 RGB
    stride: int = 16        # the source's frames_per_chunk: frame_id = chunk * stride + idx

    @property
    def frame_id(self) -> int:
        return self.chunk * self.stride + self.idx


class FrameSource(Protocol):
    fps: int                # 16
    width: int              # 832
    height: int             # 464
    frames_per_chunk: int   # 4 * chunk_size latents (16 for chunk 4, 12 for 3, 8 for 2); chunk 0 of a live
                            # rollout has 4 * (chunk_size - 1) + 1 (the first latent decodes to one frame)

    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


class ChunkQueue:
    """Bounded hand-off from a producer thread to one consumer.

    The producer never blocks: `put_chunk` drops the oldest queued chunk when
    `max_chunks` are already waiting, so a slow transport cannot stall generation.
    """

    def __init__(self, max_chunks: int = 2):
        self.max_chunks = max_chunks
        self._q: queue.Queue[list[Frame] | None] = queue.Queue()
        self._lock = threading.Lock()
        self.dropped_chunks = 0
        self.closed = False

    def put_chunk(self, frames: list[Frame]) -> None:
        with self._lock:
            while self._q.qsize() >= self.max_chunks:
                try:
                    if self._q.get_nowait() is not None:
                        self.dropped_chunks += 1
                except queue.Empty:
                    break
            self._q.put(frames)

    def close(self) -> None:
        self.closed = True
        self._q.put(None)

    def frames(self) -> Iterator[Frame]:
        while True:
            chunk = self._q.get()
            if chunk is None:
                return
            yield from chunk

    def qsize(self) -> int:
        return self._q.qsize()
