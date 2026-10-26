"""Exact Pólya-Gamma PG(1, z) draws from ``BayesLogit::rpg`` in a persistent R process.

The R process is started once per sampler and exchanges little-endian binary
messages over stdin/stdout (see ``r_bridges/pgdraw_bridge_binary.R``).  Its
RNG state can be read and restored so that checkpointed runs resume exactly.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_R_BINARY_BRIDGE = _HERE / "r_bridges" / "pgdraw_bridge_binary.R"

_INT32 = struct.Struct("<i")
_RESPONSE_HEADER = struct.Struct("<ii")


def _resolve_rscript() -> str:
    """Locate Rscript on PATH, under R_HOME, or in the default Windows install directory."""
    on_path = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if on_path is not None:
        return on_path

    r_home = os.environ.get("R_HOME")
    if r_home:
        candidate = Path(r_home) / "bin" / ("Rscript.exe" if os.name == "nt" else "Rscript")
        if candidate.exists():
            return str(candidate)

    if os.name == "nt":
        roots = [Path("C:/Program Files/R")]
        candidates = [p for root in roots for p in root.glob("R-*/bin/Rscript.exe")]
        if candidates:
            return str(sorted(candidates)[-1])

    raise FileNotFoundError(
        "Rscript is required for exact Polya-Gamma draws and K-L generation. Add Rscript to PATH "
        "or set R_HOME to the R installation directory."
    )


def bayeslogit_available() -> bool:
    """Return whether both Rscript and the BayesLogit package are available."""
    try:
        rscript = _resolve_rscript()
    except FileNotFoundError:
        return False
    probe = subprocess.run(
        [
            rscript,
            "--vanilla",
            "-e",
            (
                'local_lib <- path.expand("~/Rlibs"); '
                '.libPaths(c(local_lib, .libPaths())); '
                'quit(status=if (requireNamespace("BayesLogit", quietly=TRUE)) 0 else 1)'
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


class BayesLogitRPGSampler:
    """Wrapper around a persistent R subprocess that draws from ``BayesLogit::rpg``.

    Each `draw` call sends all tilts z as float64 and returns PG(1, z) draws of
    the same shape.  On Windows responses arrive as hexadecimal lines.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        if not _R_BINARY_BRIDGE.exists():
            raise FileNotFoundError(
                f"R bridge not found: {_R_BINARY_BRIDGE}"
            )
        rscript = _resolve_rscript()
        if not bayeslogit_available():
            raise RuntimeError(
                "R package BayesLogit is required for exact Pólya-Gamma draws; "
                "install it in ~/Rlibs or another R library path"
            )
        cmd = [
            rscript,
            str(_R_BINARY_BRIDGE),
            str(int(seed)) if seed is not None else "NA",
        ]
        self._hex_output = os.name == "nt"
        self._response_payload: bytes | None = None
        if self._hex_output:
            cmd.append("hex-output")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.backend = "bayeslogit_r"

    def _read_exact(self, size: int) -> bytes:
        if self._response_payload is not None:
            payload = self._response_payload
            self._response_payload = None
            if len(payload) != size:
                raise RuntimeError(
                    "BayesLogit bridge returned a malformed response payload "
                    f"({len(payload)} bytes, expected {size})"
                )
            return payload
        if self.proc.stdout is None:
            raise RuntimeError("BayesLogit R bridge stdout is unavailable")
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self.proc.stdout.read(size - len(chunks))
            if not chunk:
                stderr = (
                    self.proc.stderr.read().decode(errors="replace")
                    if self.proc.stderr is not None
                    else ""
                )
                raise RuntimeError(
                    "BayesLogit binary R bridge exited early.\n"
                    f"stderr:\n{stderr}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_response_header(self) -> int:
        if self._hex_output:
            assert self.proc.stdout is not None
            line = self.proc.stdout.readline()
            if not line:
                return self._read_exact(_RESPONSE_HEADER.size)
            try:
                response = bytes.fromhex(line.decode("ascii").strip())
            except (UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError(
                    "BayesLogit bridge returned invalid hexadecimal output"
                ) from exc
            if len(response) < _RESPONSE_HEADER.size:
                raise RuntimeError("BayesLogit bridge response header is truncated")
            header = response[:_RESPONSE_HEADER.size]
            self._response_payload = response[_RESPONSE_HEADER.size:]
        else:
            header = self._read_exact(_RESPONSE_HEADER.size)
        status, size = _RESPONSE_HEADER.unpack(header)
        if status != 0 or size < 0:
            raise RuntimeError(
                "BayesLogit binary R bridge returned an invalid response "
                f"(status={status}, size={size})"
            )
        if self._hex_output and size == 0 and self._response_payload == b"":
            self._response_payload = None
        return size

    def draw(self, tilt: np.ndarray) -> np.ndarray:
        original_shape = np.asarray(tilt).shape
        z = np.ascontiguousarray(tilt, dtype="<f8").ravel()
        if self.proc.poll() is not None:
            err = (
                self.proc.stderr.read().decode(errors="replace")
                if self.proc.stderr is not None
                else ""
            )
            raise RuntimeError(f"BayesLogit R bridge exited early.\nstderr:\n{err}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(_INT32.pack(1))
        self.proc.stdin.write(_INT32.pack(z.size))
        self.proc.stdin.write(z.tobytes(order="C"))
        self.proc.stdin.flush()
        size = self._read_response_header()
        if size != z.size:
            raise RuntimeError(
                f"BayesLogit R bridge returned {size} draws, expected {z.size}"
            )
        out = np.frombuffer(self._read_exact(8 * size), dtype="<f8").copy()
        return out.reshape(original_shape)

    def get_rng_state(self) -> np.ndarray:
        """Return the persistent R process's integer ``.Random.seed``."""
        if self.proc.poll() is not None:
            raise RuntimeError("BayesLogit R bridge exited before GET_STATE")
        assert self.proc.stdin is not None
        self.proc.stdin.write(_INT32.pack(2))
        self.proc.stdin.flush()
        size = self._read_response_header()
        state = np.frombuffer(
            self._read_exact(4 * size), dtype="<i4"
        ).astype(np.int64)
        if state.size == 0:
            raise RuntimeError("BayesLogit R bridge returned an empty RNG state")
        return state

    def set_rng_state(self, state: np.ndarray) -> None:
        """Restore the persistent R process's integer ``.Random.seed``."""
        values = np.asarray(state, dtype=np.int64).ravel()
        if values.size == 0:
            raise ValueError("R RNG state cannot be empty")
        if self.proc.poll() is not None:
            raise RuntimeError("BayesLogit R bridge exited before SET_STATE")
        assert self.proc.stdin is not None
        state = np.ascontiguousarray(values, dtype="<i4")
        self.proc.stdin.write(_INT32.pack(3))
        self.proc.stdin.write(_INT32.pack(state.size))
        self.proc.stdin.write(state.tobytes(order="C"))
        self.proc.stdin.flush()
        size = self._read_response_header()
        if size != 0:
            raise RuntimeError("BayesLogit RNG state restore returned payload")

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            try:
                self.proc.stdin.write(_INT32.pack(4))
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self) -> "BayesLogitRPGSampler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
