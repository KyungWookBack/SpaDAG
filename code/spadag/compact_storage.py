"""Trace storage: bounded buffers, online moments and checksum-linked chunks.

Each chunk records the name and SHA-256 of its predecessor, and only the head
link is checkpointed.  Chunks left unreferenced by an interrupted write are
kept as orphans and never replayed or overwritten.  Links hold relative paths,
so a checkpoint must be moved together with its chunk directory.  Storage
draws no random numbers from the sampler's RNG (chunk names use OS randomness).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
from contextlib import contextmanager

import numpy as np


@contextmanager
def writer_lock(directory):
    """OS-owned lock, released on process exit, including crashes."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+b") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("trace directory already has a writer") from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Moments:
    """Float64 Welford moments; SD is posterior spread, not MCSE or a CI."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def sd(self):
        if self.n < 2:
            return np.full_like(self.mean, np.nan)
        return np.sqrt(np.maximum(0, self.m2 / (self.n - 1)))


def triangle_coefficients(b, triangles):
    """Mean and root-mean-square over each triangle of the P1 surfaces ``b`` (E, H)."""
    vertices = b[:, triangles]  # E, T, 3
    total = vertices.sum(axis=-1)
    return total / 3, np.sqrt((total**2 + (vertices**2).sum(axis=-1)) / 12)


def chunk_links(directory, head):
    """Validate and return committed history in chronological order."""
    root = Path(directory)
    links, seen = [], set()
    while head is not None:
        name = head["file"]
        if Path(name).name != name or name in seen:
            raise ValueError("invalid or cyclic chunk link")
        seen.add(name)
        path = root / name
        if not path.is_file() or digest(path) != head["sha256"]:
            raise ValueError(f"missing or corrupt committed chunk: {path}")
        links.append(head)
        with np.load(path, allow_pickle=False) as z:
            head = json.loads(str(z["parent"]))
    return links[::-1]


class ChunkWriter:
    def __init__(self, directory, *, head=None, max_bytes=8 * 1024**2):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.head = head
        self.max_bytes = max_bytes
        self.buffers = {}
        self.buffer_bytes = 0
        self.peak_buffer_bytes = 0

    def add(self, stream, iteration, **values):
        row = {"iteration": np.asarray(iteration, dtype=np.int64), **values}
        size = sum(np.asarray(v).nbytes for v in row.values())
        if self.buffer_bytes and self.buffer_bytes + size > self.max_bytes:
            self.flush()
        for key, value in row.items():
            self.buffers.setdefault(f"{stream}/{key}", []).append(np.array(value, copy=True))
        self.buffer_bytes += size
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.buffer_bytes)
        if self.buffer_bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.buffers:
            return
        arrays = {k: np.stack(v) for k, v in self.buffers.items()}
        name = "chunk-" + uuid.uuid4().hex + ".npz"
        path = self.directory / name
        atomic_npz(path, parent=json.dumps(self.head), **arrays)
        self.head = {"file": name, "sha256": digest(path)}
        self.buffers.clear()
        self.buffer_bytes = 0


def iter_chunks(directory, head):
    for link in chunk_links(directory, head):
        with np.load(Path(directory) / link["file"], allow_pickle=False) as z:
            yield {k: z[k] for k in z.files if k != "parent"}


def load_stream(directory, head, stream):
    """Materialize one small analysis stream, never all full fields together."""
    columns = {}
    prefix = stream + "/"
    for chunk in iter_chunks(directory, head):
        for k, v in chunk.items():
            if k.startswith(prefix):
                columns.setdefault(k[len(prefix):], []).append(v)
    return {k: np.concatenate(v) for k, v in columns.items()}
