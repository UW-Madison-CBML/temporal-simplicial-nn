"""Electrode positions for the SEED-VII 62-channel montage.

``channel_62_pos.locs`` is the montage SEED-VII's own preprocessing feeds to
``mne.channels.read_custom_montage`` (see ``SEED-VII/src/load_cnt_file_with_mne.py``),
so its row order is the canonical SEED 62-channel order -- row ``i`` (1-based in the
file) is node index ``i - 1`` in the extracted complexes.

EEGLAB ``.locs`` polar convention, confirmed against the file's own landmarks:
``theta`` is degrees clockwise from the nose and ``radius`` is 0 at the vertex to
~0.511 at the equator, so Cz is ``(90, 0)`` (centre), Fpz ``(0, 0.511)`` (front),
Oz ``(180, 0.511)`` (back), T7 ``(-90, 0.511)`` (left) and T8 ``(90, 0.511)`` (right).
That maps to a head-centred 2D frame as::

    x = radius * sin(theta)   # +x = right
    y = radius * cos(theta)   # +y = front

A unit-sphere ``z`` is added so the coordinates keep the scalp's curvature rather
than a flattened projection: ``radius`` is treated as a polar arc angle scaled so the
equator (0.511) sits at the horizon.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent

DEFAULT_LOCS = _ROOT / "data" / "channel_62_pos.locs"
N_CHANNELS = 62
# radius of the equator ring in the .locs file (T7/T8/Fpz/Oz all sit here)
_EQUATOR_RADIUS = 0.51111


def load_locs(path: Path | None = None) -> tuple[list[str], torch.Tensor]:
    """Parse a ``.locs`` montage into ``(labels, polar)`` with ``polar = (N, 2)``.

    Columns are ``index theta radius label``, whitespace separated (the file mixes
    tabs and spaces, so split on generic whitespace).
    """
    path = Path(path) if path is not None else DEFAULT_LOCS
    labels: list[str] = []
    rows: list[tuple[float, float]] = []
    with path.open() as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed .locs line in {path}: {raw_line!r}")
            _idx, theta, radius, label = parts[0], parts[1], parts[2], parts[3]
            labels.append(label)
            rows.append((float(theta), float(radius)))
    if not rows:
        raise ValueError(f"No channel rows parsed from {path}")
    return labels, torch.tensor(rows, dtype=torch.float32)


def polar_to_cartesian(polar: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, 2)`` ``(theta_deg, radius)`` to ``(N, 3)`` ``(x, y, z)``.

    ``x``/``y`` follow the head-centred frame described in the module docstring; ``z``
    lifts the montage back onto a unit hemisphere so that, e.g., Cz sits above the
    ring of temporal electrodes instead of collapsing onto the same plane.
    """
    theta = torch.deg2rad(polar[:, 0])
    radius = polar[:, 1]
    x = radius * torch.sin(theta)
    y = radius * torch.cos(theta)
    # arc angle from the vertex: 0 at Cz, pi/2 at the equator ring
    arc = (radius / _EQUATOR_RADIUS).clamp(min=0.0) * (math.pi / 2.0)
    z = torch.cos(arc) * _EQUATOR_RADIUS
    return torch.stack([x, y, z], dim=-1)


def fourier_features(coords: torch.Tensor, n_bands: int) -> torch.Tensor:
    """Append ``sin``/``cos`` of geometrically-spaced frequencies of ``coords``.

    Raw 3-d coordinates are a very low-dimensional signal for a linear layer to work
    with; lifting them into a Fourier basis (as in NeRF-style positional encodings)
    lets the model separate nearby electrodes. Returns
    ``(N, D * (1 + 2 * n_bands))`` -- the raw coordinates followed by each band.
    """
    if n_bands <= 0:
        return coords
    freqs = 2.0 ** torch.arange(n_bands, dtype=coords.dtype, device=coords.device)
    scaled = coords.unsqueeze(-1) * freqs * math.pi  # (N, D, n_bands)
    sin = torch.sin(scaled).flatten(start_dim=1)
    cos = torch.cos(scaled).flatten(start_dim=1)
    return torch.cat([coords, sin, cos], dim=-1)


def channel_position_features(
    path: Path | None = None,
    *,
    n_fourier_bands: int = 4,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the static per-channel position features, shape ``(62, D)``.

    Parameters
    ----------
    path : Path, optional
        ``.locs`` montage file; defaults to the copy bundled under ``data/``.
    n_fourier_bands : int, default=4
        Fourier bands to lift the raw coordinates into (0 = raw ``(x, y, z)`` only).
    normalize : bool, default=True
        Scale raw coordinates to roughly unit range before encoding.
    """
    labels, polar = load_locs(path)
    if len(labels) != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} channels in the montage, parsed {len(labels)}"
        )
    coords = polar_to_cartesian(polar)
    if normalize:
        coords = coords / _EQUATOR_RADIUS
    return fourier_features(coords, n_fourier_bands)
