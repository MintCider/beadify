"""拼豆图纸生成器 - Beadify

Converts pixel art images into bead pattern sheets with color codes.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pillow",
# ]
# ///

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASIC_PREFIXES = set("ABCDEFGHM")
CELL_SIZE = 40
FONT_SIZE = 12
GRID_LINE_WIDTH = 1
DEFAULT_TOLERANCE = 10.0
MIN_USAGE_THRESHOLD = 3
BG_COLOR = (210, 210, 210)  # background for empty areas (darker to contrast with white beads)

# ---------------------------------------------------------------------------
# Color conversion: sRGB -> CIE Lab (pure numpy)
# ---------------------------------------------------------------------------

# sRGB to XYZ (D65) matrix
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# D65 reference white
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0-255) to CIE Lab. Input shape: (..., 3)."""
    c = rgb.astype(np.float64) / 255.0
    # Gamma decode
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    # Linear RGB -> XYZ
    xyz = linear @ _SRGB_TO_XYZ.T
    # Normalize by D65 white
    xyz = xyz / _D65_WHITE
    # XYZ -> Lab (piecewise)
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3,
                 np.cbrt(xyz),
                 xyz / (3.0 * delta**2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference. Inputs: (..., 3) Lab arrays. Returns (...)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg**7
    G = 0.5 * (1.0 - np.sqrt(C_avg7 / (C_avg7 + 25.0**7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = np.where(
        (C1p * C2p) == 0, 0.0,
        np.where(np.abs(h2p - h1p) <= 180, h2p - h1p,
                 np.where(h2p - h1p > 180, h2p - h1p - 360,
                          h2p - h1p + 360)))
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    hp_avg = np.where(
        (C1p * C2p) == 0, h1p + h2p,
        np.where(np.abs(h1p - h2p) <= 180, (h1p + h2p) / 2.0,
                 np.where(h1p + h2p < 360,
                          (h1p + h2p + 360) / 2.0,
                          (h1p + h2p - 360) / 2.0)))

    T = (1.0
         - 0.17 * np.cos(np.radians(hp_avg - 30))
         + 0.24 * np.cos(np.radians(2 * hp_avg))
         + 0.32 * np.cos(np.radians(3 * hp_avg + 6))
         - 0.20 * np.cos(np.radians(4 * hp_avg - 63)))

    SL = 1.0 + 0.015 * (Lp_avg - 50)**2 / np.sqrt(20 + (Lp_avg - 50)**2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg**7
    RC = 2.0 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25.0**7))
    d_theta = 30.0 * np.exp(-((hp_avg - 275) / 25.0)**2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    return np.sqrt(
        (dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2
        + RT * (dCp / SC) * (dHp / SH))


def lab_euclidean(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 color difference (Euclidean in Lab). Same interface as ciede2000."""
    return np.sqrt(((lab1 - lab2) ** 2).sum(axis=-1))


def color_distance(lab1: np.ndarray, lab2: np.ndarray,
                   metric: str = "cie76") -> np.ndarray:
    """Compute color distance. metric: 'cie76', 'ciede2000', or 'hybrid'."""
    if metric == "ciede2000":
        return ciede2000(lab1, lab2)
    if metric == "hybrid":
        return np.sqrt(lab_euclidean(lab1, lab2) * ciede2000(lab1, lab2))
    return lab_euclidean(lab1, lab2)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (R, G, B)."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Bead color database
# ---------------------------------------------------------------------------

def load_bead_colors(
    json_path: str, allow_extended: bool
) -> tuple[np.ndarray, list[str], dict[str, tuple[int, int, int]]]:
    """Load bead colors, returning (bead_lab, bead_labels, label_to_rgb).

    If allow_extended is False, only A-M series are kept.
    Transparent beads are always excluded.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transparent = set(data.get("transparent", []))
    label_to_hex = data["label_to_hex"]

    labels: list[str] = []
    rgbs: list[tuple[int, int, int]] = []
    for label, hexval in label_to_hex.items():
        if label in transparent:
            continue
        # Extract alphabetic prefix
        prefix = ""
        for ch in label:
            if ch.isalpha():
                prefix += ch
            else:
                break
        if not allow_extended and prefix not in BASIC_PREFIXES:
            continue
        labels.append(label)
        rgbs.append(hex_to_rgb(hexval))

    rgb_arr = np.array(rgbs, dtype=np.uint8)
    lab_arr = rgb_to_lab(rgb_arr)
    label_to_rgb = {label: rgb for label, rgb in zip(labels, rgbs)}
    return lab_arr, labels, label_to_rgb


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------

def _gradient_profile(img: np.ndarray, axis: int) -> np.ndarray:
    """Compute 1D gradient magnitude profile along the given axis.

    axis=1 -> horizontal gradient (column boundaries), profile length = W
    axis=0 -> vertical gradient (row boundaries), profile length = H
    """
    rgba = img.astype(np.float64)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    if axis == 1:
        diff = np.diff(rgb, axis=1)  # (H, W-1, 3)
        grad = np.sqrt((diff**2).sum(axis=2))  # (H, W-1)
        # Zero out where either neighbor is transparent
        mask = (alpha[:, :-1] > 128) & (alpha[:, 1:] > 128)
        grad *= mask
        profile = grad.sum(axis=0)  # (W-1,)
        # Pad to length W
        profile = np.concatenate([[0], profile])
    else:
        diff = np.diff(rgb, axis=0)  # (H-1, W, 3)
        grad = np.sqrt((diff**2).sum(axis=2))  # (H-1, W)
        mask = (alpha[:-1, :] > 128) & (alpha[1:, :] > 128)
        grad *= mask
        profile = grad.sum(axis=1)  # (H-1,)
        profile = np.concatenate([[0], profile])
    return profile


def _autocorrelation_period(profile: np.ndarray, min_period: int = 5,
                            max_period: int = 100) -> int:
    """Find the dominant period in a 1D signal via FFT autocorrelation."""
    signal = profile - profile.mean()
    n = len(signal)
    fft = np.fft.rfft(signal, n=2 * n)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    if acf[0] == 0:
        raise ValueError("Gradient profile is flat — cannot detect block size")
    acf = acf / acf[0]

    # Find first significant peak after lag=min_period
    search = acf[min_period:max_period + 1]
    if len(search) == 0:
        raise ValueError("Cannot detect block size in the given range")

    # Find local maxima
    peaks: list[tuple[int, float]] = []
    for i in range(1, len(search) - 1):
        if search[i] > search[i - 1] and search[i] > search[i + 1]:
            peaks.append((i + min_period, float(search[i])))

    if not peaks:
        raise ValueError("No autocorrelation peak found")

    # Return the lag of the strongest peak
    best_lag, best_val = max(peaks, key=lambda x: x[1])
    if best_val < 0.2:
        raise ValueError(f"Weak autocorrelation peak ({best_val:.2f})")

    # Prefer fundamental frequency over harmonics: when adjacent blocks share
    # the same color the gradient profile has gaps, boosting harmonics above
    # the true period.  If a smaller lag divides the strongest lag and is
    # itself reasonably strong, it is the real block size.
    for lag, val in sorted(peaks, key=lambda x: x[0]):
        if lag >= best_lag:
            break
        if val < 0.3 or val < best_val * 0.4:
            continue
        ratio = best_lag / lag
        nearest_int = round(ratio)
        if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < 0.15:
            return lag

    return best_lag


def detect_block_size(img: np.ndarray, override: int = 0) -> int:
    """Detect the pixel art block size. If override > 0, use that directly."""
    if override > 0:
        return override
    prof_h = _gradient_profile(img, axis=1)
    prof_v = _gradient_profile(img, axis=0)
    try:
        bs_h = _autocorrelation_period(prof_h)
    except ValueError:
        bs_h = 0
    try:
        bs_v = _autocorrelation_period(prof_v)
    except ValueError:
        bs_v = 0

    if bs_h > 0 and bs_v > 0:
        # They should be close; prefer the average if within 20%
        if abs(bs_h - bs_v) / max(bs_h, bs_v) < 0.2:
            return round((bs_h + bs_v) / 2)
        # Otherwise pick the one with stronger signal
        peak_h = _gradient_profile(img, 1).max()
        peak_v = _gradient_profile(img, 0).max()
        return bs_h if peak_h >= peak_v else bs_v
    elif bs_h > 0:
        return bs_h
    elif bs_v > 0:
        return bs_v
    else:
        raise ValueError("Cannot auto-detect block size. Use -b to specify.")


def _find_boundaries(profile: np.ndarray, block_size: int) -> list[int]:
    """Find actual block boundary positions using gradient peaks.

    Returns a sorted list of pixel positions where block boundaries are.
    """
    # Smooth the profile slightly to reduce noise
    kernel_size = max(3, block_size // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(profile, kernel, mode="same")

    # Find the first significant peak to anchor the grid
    threshold = smoothed.max() * 0.15
    search_radius = int(block_size * 0.3)

    # Collect all local maxima above threshold
    all_peaks: list[int] = []
    for i in range(1, len(smoothed) - 1):
        if (smoothed[i] > smoothed[i - 1] and
                smoothed[i] > smoothed[i + 1] and
                smoothed[i] > threshold):
            all_peaks.append(i)

    if not all_peaks:
        # Fallback: uniform grid
        return list(range(0, len(profile), block_size))

    def _snap_walk(start: int, direction: int, radius: int) -> list[int]:
        """Walk from start, snapping to nearest peak within radius."""
        result: list[int] = []
        pos = start
        while True:
            nxt = pos + direction * block_size
            if nxt < 0 or nxt >= len(profile):
                break
            lo = max(0, nxt - radius)
            hi = min(len(smoothed), nxt + radius + 1)
            w = smoothed[lo:hi]
            if w.max() > threshold:
                cand = lo + int(np.argmax(w))
                if (direction > 0 and cand > pos) or \
                   (direction < 0 and cand < pos):
                    pos = cand
                else:
                    pos = nxt
            else:
                pos = nxt
            result.append(pos)
        return result

    # --- Initial greedy pass ---
    first_peak = all_peaks[0]
    left = _snap_walk(first_peak, -1, search_radius)
    left.reverse()
    right = _snap_walk(first_peak, +1, search_radius)
    boundaries = left + [first_peak] + right
    boundaries = sorted(set(boundaries))

    # --- Post-processing: fix drift from missed boundaries ---
    # When a boundary lands on a non-peak (no gradient support), subsequent
    # boundaries may drift. Re-anchor from the nearest strong peak using a
    # wider search radius to correct the chain.
    wide_radius = int(block_size * 0.6)
    peak_set = set(all_peaks)

    for _pass in range(3):
        changed = False
        for i in range(len(boundaries)):
            # Check if this boundary has gradient support
            b = boundaries[i]
            has_support = any(abs(b - p) <= 2 for p in peak_set)
            if has_support:
                continue
            # This boundary was guessed — find the nearest strong peak
            # within a wider radius and re-walk from there
            lo = max(0, b - wide_radius)
            hi = min(len(smoothed), b + wide_radius + 1)
            w = smoothed[lo:hi]
            if w.max() <= threshold:
                continue  # no peak nearby at all, keep the guess
            anchor = lo + int(np.argmax(w))
            if anchor == b:
                continue
            # Re-walk forward from anchor
            new_tail = _snap_walk(anchor, +1, wide_radius)
            # Replace boundaries from i onward
            boundaries = boundaries[:i] + [anchor] + new_tail
            boundaries = sorted(set(boundaries))
            changed = True
            break  # restart scan
        if not changed:
            break

    # Add image edges as implicit boundaries
    if boundaries[0] > block_size * 0.5:
        boundaries.insert(0, 0)
    if len(profile) - boundaries[-1] > block_size * 0.5:
        boundaries.append(len(profile))

    return boundaries


def detect_grid(img: np.ndarray, block_size: int) -> tuple[list[int], list[int]]:
    """Detect grid boundaries. Returns (row_bounds, col_bounds)."""
    prof_h = _gradient_profile(img, axis=1)
    prof_v = _gradient_profile(img, axis=0)
    col_bounds = _find_boundaries(prof_h, block_size)
    row_bounds = _find_boundaries(prof_v, block_size)
    return row_bounds, col_bounds


# ---------------------------------------------------------------------------
# Block color extraction
# ---------------------------------------------------------------------------

def extract_block_colors(
    img: np.ndarray, row_bounds: list[int], col_bounds: list[int],
    sample_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean color of each block's center region in Lab space.

    Args:
        sample_ratio: fraction of each block (centered) to sample, 0.0-1.0.
            0.5 = center 50%, 1.0 = full block.

    Returns:
        grid_lab: (n_rows, n_cols, 3) float64 array of median Lab per block
        grid_mask: (n_rows, n_cols) bool array, True if block is opaque
    """
    sample_ratio = max(0.1, min(1.0, sample_ratio))
    margin_frac = (1.0 - sample_ratio) / 2.0

    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    grid_lab = np.zeros((n_rows, n_cols, 3), dtype=np.float64)
    grid_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for r in range(n_rows):
        y0, y1 = row_bounds[r], row_bounds[r + 1]
        bh = y1 - y0
        my = int(bh * margin_frac)
        cy0, cy1 = y0 + my, y1 - my
        if cy1 <= cy0:
            cy0, cy1 = y0, y1

        for c in range(n_cols):
            x0, x1 = col_bounds[c], col_bounds[c + 1]
            bw = x1 - x0
            mx = int(bw * margin_frac)
            cx0, cx1 = x0 + mx, x1 - mx
            if cx1 <= cx0:
                cx0, cx1 = x0, x1

            block = img[cy0:cy1, cx0:cx1]
            alpha = block[..., 3]
            opaque = alpha > 128

            if opaque.sum() < max(1, opaque.size * 0.3):
                # Mostly transparent
                grid_mask[r, c] = False
                continue

            pixels_rgb = block[opaque][..., :3]  # (N, 3)
            pixels_lab = rgb_to_lab(pixels_rgb)
            grid_lab[r, c] = np.mean(pixels_lab, axis=0)
            grid_mask[r, c] = True

    # Crop to bounding box of content
    rows_with_content = np.any(grid_mask, axis=1)
    cols_with_content = np.any(grid_mask, axis=0)
    if not rows_with_content.any():
        return grid_lab, grid_mask

    r_min, r_max = np.where(rows_with_content)[0][[0, -1]]
    c_min, c_max = np.where(cols_with_content)[0][[0, -1]]
    grid_lab = grid_lab[r_min:r_max + 1, c_min:c_max + 1]
    grid_mask = grid_mask[r_min:r_max + 1, c_min:c_max + 1]
    return grid_lab, grid_mask


# ---------------------------------------------------------------------------
# Color mapping & consolidation
# ---------------------------------------------------------------------------

def map_to_bead_colors(
    grid_lab: np.ndarray, grid_mask: np.ndarray,
    bead_lab: np.ndarray, bead_labels: list[str],
    metric: str = "cie76",
) -> tuple[np.ndarray, np.ndarray]:
    """Map each block to the nearest bead color in Lab space.

    Returns:
        grid_indices: (n_rows, n_cols) int array of indices into bead_labels
        grid_lab: (n_rows, n_cols, 3) Lab values (passed through)
    """
    n_rows, n_cols = grid_mask.shape

    # Flatten opaque blocks for vectorized distance computation
    flat_lab = grid_lab[grid_mask]  # (N, 3)
    dists = color_distance(flat_lab[:, np.newaxis, :],
                           bead_lab[np.newaxis, :, :], metric)
    flat_indices = dists.argmin(axis=1)

    grid_indices = np.full((n_rows, n_cols), -1, dtype=np.int32)
    grid_indices[grid_mask] = flat_indices
    return grid_indices, grid_lab


def consolidate_colors(
    grid_indices: np.ndarray, grid_mask: np.ndarray,
    grid_lab: np.ndarray, bead_lab: np.ndarray,
    bead_labels: list[str], tolerance: float,
    metric: str = "cie76",
) -> np.ndarray:
    """Three-pass color consolidation to reduce palette size."""
    from collections import Counter

    _dist = lambda a, b: float(color_distance(a, b, metric)[0])

    indices = grid_indices.copy()
    n_rows, n_cols = indices.shape

    # --- Pass 1: Rare color merging ---
    for _ in range(5):  # iterate a few times
        counts: dict[int, int] = {}
        for idx in indices[grid_mask]:
            counts[idx] = counts.get(idx, 0) + 1

        merged_any = False
        frequent = {idx for idx, cnt in counts.items() if cnt >= MIN_USAGE_THRESHOLD}
        if not frequent:
            break

        for rare_idx, cnt in list(counts.items()):
            if cnt >= MIN_USAGE_THRESHOLD:
                continue
            # Find closest frequent color
            best_freq_idx = -1
            best_dist = float("inf")
            for f_idx in frequent:
                d = _dist(bead_lab[rare_idx:rare_idx+1], bead_lab[f_idx:f_idx+1])
                if d < best_dist:
                    best_dist = d
                    best_freq_idx = f_idx
            if best_freq_idx >= 0 and best_dist < tolerance:
                indices[indices == rare_idx] = best_freq_idx
                merged_any = True

        if not merged_any:
            break

    # --- Pass 2: Global similar-color merging ---
    used_indices = sorted(set(indices[grid_mask].tolist()))
    threshold2 = tolerance * 0.7

    for i in range(len(used_indices)):
        for j in range(i + 1, len(used_indices)):
            idx_a, idx_b = used_indices[i], used_indices[j]
            mask_a = indices == idx_a
            mask_b = indices == idx_b
            cnt_a = int(mask_a[grid_mask].sum())
            cnt_b = int(mask_b[grid_mask].sum())
            if cnt_a == 0 or cnt_b == 0:
                continue
            d = _dist(bead_lab[idx_a:idx_a+1], bead_lab[idx_b:idx_b+1])
            if d < threshold2:
                if cnt_a >= cnt_b:
                    indices[mask_b & grid_mask] = idx_a
                else:
                    indices[mask_a & grid_mask] = idx_b

    # --- Pass 3: Local neighbor smoothing ---
    threshold3 = tolerance * 0.5
    for _iteration in range(3):
        changed = False
        for r in range(n_rows):
            for c in range(n_cols):
                if not grid_mask[r, c]:
                    continue
                cur_idx = int(indices[r, c])
                neighbors: list[int] = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and grid_mask[nr, nc]:
                        neighbors.append(int(indices[nr, nc]))
                if not neighbors:
                    continue

                neighbor_counts = Counter(neighbors)
                most_common_idx, most_common_cnt = neighbor_counts.most_common(1)[0]
                if most_common_idx == cur_idx or most_common_cnt < 2:
                    continue

                orig_lab_pt = grid_lab[r, c:c+1]
                cur_dist = _dist(orig_lab_pt, bead_lab[cur_idx:cur_idx+1])
                new_dist = _dist(orig_lab_pt, bead_lab[most_common_idx:most_common_idx+1])

                if new_dist - cur_dist < threshold3 and new_dist < tolerance * 1.5:
                    indices[r, c] = most_common_idx
                    changed = True
        if not changed:
            break

    return indices


# ---------------------------------------------------------------------------
# Connected components, border, and connection
# ---------------------------------------------------------------------------

def find_connected_components(mask: np.ndarray) -> np.ndarray:
    """4-connected component labeling via union-find. Returns (H, W) int array."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    parent: dict[int, int] = {}
    next_label = 1

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            neighbors = []
            if r > 0 and mask[r - 1, c]:
                neighbors.append(labels[r - 1, c])
            if c > 0 and mask[r, c - 1]:
                neighbors.append(labels[r, c - 1])
            if not neighbors:
                labels[r, c] = next_label
                parent[next_label] = next_label
                next_label += 1
            else:
                min_lbl = min(neighbors)
                labels[r, c] = min_lbl
                for n in neighbors:
                    union(n, min_lbl)

    # Flatten
    for r in range(H):
        for c in range(W):
            if labels[r, c] > 0:
                labels[r, c] = find(labels[r, c])

    # Renumber to 1..K
    unique = sorted(set(labels[labels > 0]))
    remap = {old: new + 1 for new, old in enumerate(unique)}
    for r in range(H):
        for c in range(W):
            if labels[r, c] > 0:
                labels[r, c] = remap[labels[r, c]]
    return labels


def add_border(
    grid_indices: np.ndarray, grid_mask: np.ndarray,
    bead_labels: list[str], border_label: str = "H2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add 1-cell border around opaque bodies. Expands grid by 1 on each side.

    Returns (new_indices, new_mask, border_flag) all shaped (H+2, W+2).
    border_flag[r,c] is True for border cells (not original content).
    """
    H, W = grid_mask.shape
    padded_mask = np.zeros((H + 2, W + 2), dtype=bool)
    padded_mask[1:-1, 1:-1] = grid_mask
    padded_indices = np.full((H + 2, W + 2), -1, dtype=np.int32)
    padded_indices[1:-1, 1:-1] = grid_indices

    # 4-connected dilation via shifts
    dilated = np.zeros_like(padded_mask)
    dilated[1:, :] |= padded_mask[:-1, :]
    dilated[:-1, :] |= padded_mask[1:, :]
    dilated[:, 1:] |= padded_mask[:, :-1]
    dilated[:, :-1] |= padded_mask[:, 1:]

    border_cells = dilated & ~padded_mask
    if border_label in bead_labels:
        border_idx = bead_labels.index(border_label)
    else:
        border_idx = len(bead_labels)
        bead_labels = list(bead_labels) + [border_label]
    padded_indices[border_cells] = border_idx
    padded_mask[border_cells] = True

    border_flag = np.zeros_like(padded_mask)
    border_flag[border_cells] = True
    return padded_indices, padded_mask, border_flag


def _rasterize_4connected(r1: int, c1: int, r2: int, c2: int) -> list[tuple[int, int]]:
    """Rasterize a line as 4-connected grid cells (Bresenham variant).

    Every consecutive pair of cells shares an edge (not just a corner).
    Total steps = |dr| + |dc|.
    """
    path = [(r1, c1)]
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r2 > r1 else (-1 if r2 < r1 else 0)
    sc = 1 if c2 > c1 else (-1 if c2 < c1 else 0)
    r, c = r1, c1

    if dc >= dr:
        err = dc // 2
        for _ in range(dc):
            err -= dr
            if err < 0:
                r += sr
                path.append((r, c))
                err += dc
            c += sc
            path.append((r, c))
    else:
        err = dr // 2
        for _ in range(dr):
            err -= dc
            if err < 0:
                c += sc
                path.append((r, c))
                err += dr
            r += sr
            path.append((r, c))
    return path


def connect_bodies(
    grid_indices: np.ndarray, grid_mask: np.ndarray,
    bead_labels: list[str], connect_label: str = "H1",
    width: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Connect disconnected bodies via 4-connected rasterized lines.

    Uses Bresenham 4-connected variant for any angle. Width is constrained
    by the overlap of both bodies' projections perpendicular to the
    connection direction. Each lane extends from body A to body B.
    Falls back to single-width line when no perpendicular overlap exists.

    Returns (new_indices, new_mask, connect_flag) same shape as input.
    """
    from collections import defaultdict

    H, W = grid_mask.shape
    comp_labels = find_connected_components(grid_mask)
    n_comps = int(comp_labels.max())
    if n_comps <= 1:
        return (grid_indices.copy(), grid_mask.copy(),
                np.zeros_like(grid_mask))

    comp_cells: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if comp_labels[r, c] > 0:
                comp_cells[comp_labels[r, c]].append((r, c))

    comp_ids = sorted(comp_cells.keys())

    # Closest cell pair between each component pair (Manhattan distance)
    edges: list[tuple[int, int, int, tuple[int, int], tuple[int, int]]] = []
    for i in range(len(comp_ids)):
        for j in range(i + 1, len(comp_ids)):
            a, b = comp_ids[i], comp_ids[j]
            ca = np.array(comp_cells[a])
            cb = np.array(comp_cells[b])
            best_dist = float("inf")
            best_pair = (ca[0].tolist(), cb[0].tolist())
            for chunk_start in range(0, len(ca), 500):
                chunk = ca[chunk_start:chunk_start + 500]
                dists = (np.abs(chunk[:, None, 0] - cb[None, :, 0])
                         + np.abs(chunk[:, None, 1] - cb[None, :, 1]))
                idx = np.unravel_index(int(dists.argmin()), dists.shape)
                d = int(dists[idx])
                if d < best_dist:
                    best_dist = d
                    best_pair = (tuple(chunk[idx[0]]), tuple(cb[idx[1]]))
            edges.append((int(best_dist), a, b, best_pair[0], best_pair[1]))

    # Kruskal MST
    edges.sort()
    uf_parent = {cid: cid for cid in comp_ids}

    def uf_find(x: int) -> int:
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    mst_edges: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    for _dist, a, b, cell_a, cell_b in edges:
        if uf_find(a) != uf_find(b):
            uf_parent[uf_find(a)] = uf_find(b)
            mst_edges.append((cell_a, cell_b, a, b))

    if connect_label in bead_labels:
        connect_label_idx = bead_labels.index(connect_label)
    else:
        connect_label_idx = len(bead_labels)
        bead_labels = list(bead_labels) + [connect_label]
    new_indices = grid_indices.copy()
    new_mask = grid_mask.copy()
    connect_flag = np.zeros_like(grid_mask)

    def _place(r: int, c: int) -> None:
        if 0 <= r < H and 0 <= c < W and not new_mask[r, c]:
            new_indices[r, c] = connect_label_idx
            new_mask[r, c] = True
            connect_flag[r, c] = True

    for cell_a, cell_b, comp_a, comp_b in mst_edges:
        r1, c1 = cell_a
        r2, c2 = cell_b
        cells_a = comp_cells[comp_a]
        cells_b = comp_cells[comp_b]

        # Perpendicular axis: rows if |dc|>=|dr|, columns otherwise
        is_horiz = abs(c2 - c1) >= abs(r2 - r1)

        if is_horiz:
            # Lanes = rows
            rows_a: dict[int, list[int]] = defaultdict(list)
            for r, c in cells_a:
                rows_a[r].append(c)
            rows_b: dict[int, list[int]] = defaultdict(list)
            for r, c in cells_b:
                rows_b[r].append(c)

            overlap = sorted(set(rows_a) & set(rows_b))
            if not overlap:
                for r, c in _rasterize_4connected(r1, c1, r2, c2):
                    _place(r, c)
                continue

            actual_w = min(width, len(overlap))
            center = (r1 + r2) // 2
            best_si = 0
            best_d = float("inf")
            for si in range(len(overlap) - actual_w + 1):
                mid = (overlap[si] + overlap[si + actual_w - 1]) / 2
                d = abs(mid - center)
                if d < best_d:
                    best_d = d
                    best_si = si
            selected = overlap[best_si:best_si + actual_w]

            # Determine left vs right body
            mean_c_a = sum(c for _, c in cells_a) / len(cells_a)
            mean_c_b = sum(c for _, c in cells_b) / len(cells_b)
            if mean_c_a <= mean_c_b:
                left_rows, right_rows = rows_a, rows_b
            else:
                left_rows, right_rows = rows_b, rows_a

            for lane_r in selected:
                lc = max(left_rows[lane_r]) + 1
                rc = min(right_rows[lane_r]) - 1
                if lc > rc:
                    continue
                for r, c in _rasterize_4connected(lane_r, lc, lane_r, rc):
                    _place(r, c)

        else:
            # Lanes = columns
            cols_a: dict[int, list[int]] = defaultdict(list)
            for r, c in cells_a:
                cols_a[c].append(r)
            cols_b: dict[int, list[int]] = defaultdict(list)
            for r, c in cells_b:
                cols_b[c].append(r)

            overlap = sorted(set(cols_a) & set(cols_b))
            if not overlap:
                for r, c in _rasterize_4connected(r1, c1, r2, c2):
                    _place(r, c)
                continue

            actual_w = min(width, len(overlap))
            center = (c1 + c2) // 2
            best_si = 0
            best_d = float("inf")
            for si in range(len(overlap) - actual_w + 1):
                mid = (overlap[si] + overlap[si + actual_w - 1]) / 2
                d = abs(mid - center)
                if d < best_d:
                    best_d = d
                    best_si = si
            selected = overlap[best_si:best_si + actual_w]

            mean_r_a = sum(r for r, _ in cells_a) / len(cells_a)
            mean_r_b = sum(r for r, _ in cells_b) / len(cells_b)
            if mean_r_a <= mean_r_b:
                top_cols, bot_cols = cols_a, cols_b
            else:
                top_cols, bot_cols = cols_b, cols_a

            for lane_c in selected:
                tr = max(top_cols[lane_c]) + 1
                br = min(bot_cols[lane_c]) - 1
                if tr > br:
                    continue
                for r, c in _rasterize_4connected(tr, lane_c, br, lane_c):
                    _place(r, c)

    return new_indices, new_mask, connect_flag


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a monospace system font, fall back to default."""
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _load_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a font with CJK glyph support, fall back to _load_font."""
    candidates = [
        # macOS
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Windows
        "msyh.ttc",    # Microsoft YaHei
        "simhei.ttf",  # SimHei
        "simsun.ttc",  # SimSun
        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return _load_font(size)


def _text_color(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Choose black or white text for contrast against the given background."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _draw_checkerboard(draw: ImageDraw.Draw, x0: int, y0: int,
                       x1: int, y1: int, cell_size: int) -> None:
    """Draw a checkerboard pattern in the given rectangle."""
    step = max(4, cell_size // 8)
    for py in range(y0, y1, step):
        for px in range(x0, x1, step):
            shade = 195 if ((px - x0 + py - y0) // step) % 2 == 0 else 215
            draw.rectangle(
                [px, py, min(px + step - 1, x1), min(py + step - 1, y1)],
                fill=(shade, shade, shade))


def _draw_cell(draw: ImageDraw.Draw, x0: int, y0: int, x1: int, y1: int,
               rgb: tuple[int, int, int], is_transparent_bead: bool,
               cell_size: int) -> None:
    """Draw a single cell, with semi-transparent checkerboard for transparent beads."""
    if is_transparent_bead:
        # Checkerboard base + semi-transparent color overlay
        _draw_checkerboard(draw, x0, y0, x1, y1, cell_size)
        # Blend: 40% bead color + 60% existing (approximate with lighter color)
        blended = tuple(int(rgb[i] * 0.35 + BG_COLOR[i] * 0.65) for i in range(3))
        # Use a PIL overlay for proper alpha blending
        draw.rectangle([x0, y0, x1, y1], fill=blended)  # type: ignore[arg-type]
    else:
        draw.rectangle([x0, y0, x1, y1], fill=rgb)


def get_pattern_margins(n_rows: int, n_cols: int, cell_size: int) -> tuple[int, int]:
    """Return (margin_lr, margin_tb) used by render_output for coordinate labels."""
    coord_font_size = max(9, cell_size // 4)
    coord_font = _load_font(coord_font_size)
    max_label = str(max(n_rows, n_cols))
    bbox_test = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), max_label, font=coord_font)
    label_w = bbox_test[2] - bbox_test[0]
    label_h = bbox_test[3] - bbox_test[1]
    return int(label_w + 8), int(label_h + 6)


def render_merge_highlight(
    img: Image.Image,
    indices_base: np.ndarray, indices_high: np.ndarray,
    grid_mask: np.ndarray, cell_size: int,
    ox: int = 0, oy: int = 0,
    n_hue_cycles: int = 3,
) -> Image.Image:
    """Draw continuous rainbow gradient borders around clusters of merged cells.

    Each border pixel gets its own hue based on angle from the cluster centroid,
    cycling through the full spectrum *n_hue_cycles* times per revolution.
    """

    diff_mask = (indices_base != indices_high) & grid_mask
    if not diff_mask.any():
        return img

    clusters = find_connected_components(diff_mask)
    n_clusters = int(clusters.max())
    border_w = max(2, cell_size // 15)
    H, W = diff_mask.shape
    img_h, img_w = img.size[1], img.size[0]

    # Work on numpy array for pixel-level drawing
    img_arr = np.array(img)

    for cid in range(1, n_clusters + 1):
        cmask = clusters == cid

        # Centroid in pixel coordinates
        rows_c, cols_c = np.where(cmask)
        cr_px = float(rows_c.mean()) * cell_size + cell_size / 2.0 + oy
        cc_px = float(cols_c.mean()) * cell_size + cell_size / 2.0 + ox

        # Build pixel-level border mask
        border_px = np.zeros((img_h, img_w), dtype=bool)

        for r in range(H):
            for c in range(W):
                if not cmask[r, c]:
                    continue
                x0 = ox + c * cell_size
                y0 = oy + r * cell_size
                x1 = min(x0 + cell_size, img_w)
                y1 = min(y0 + cell_size, img_h)
                x0 = max(x0, 0)
                y0 = max(y0, 0)

                if r == 0 or not cmask[r - 1, c]:
                    border_px[y0:min(y0 + border_w, img_h), x0:x1] = True
                if r == H - 1 or not cmask[r + 1, c]:
                    border_px[max(y1 - border_w, 0):y1, x0:x1] = True
                if c == 0 or not cmask[r, c - 1]:
                    border_px[y0:y1, x0:min(x0 + border_w, img_w)] = True
                if c == W - 1 or not cmask[r, c + 1]:
                    border_px[y0:y1, max(x1 - border_w, 0):x1] = True

        # Get pixel coordinates
        py, px = np.where(border_px)
        if len(py) == 0:
            continue

        # Angle from centroid → hue with multiple cycles
        angles = np.arctan2(py.astype(float) - cr_px,
                            px.astype(float) - cc_px)  # -pi..pi
        hues = ((angles / (2 * np.pi) + 0.5) * n_hue_cycles) % 1.0

        # Vectorized HSV→RGB  (S=0.9, V=1.0)
        c_chroma = 0.9
        m = 0.1  # V - C
        h6 = hues * 6.0
        x_val = c_chroma * (1.0 - np.abs(h6 % 2.0 - 1.0))
        sector = np.floor(h6).astype(int) % 6

        r_arr = np.full(len(py), m)
        g_arr = np.full(len(py), m)
        b_arr = np.full(len(py), m)

        # Sector 0: R=C, G=X
        s0 = sector == 0; r_arr[s0] += c_chroma; g_arr[s0] += x_val[s0]
        # Sector 1: R=X, G=C
        s1 = sector == 1; r_arr[s1] += x_val[s1]; g_arr[s1] += c_chroma
        # Sector 2: G=C, B=X
        s2 = sector == 2; g_arr[s2] += c_chroma; b_arr[s2] += x_val[s2]
        # Sector 3: G=X, B=C
        s3 = sector == 3; g_arr[s3] += x_val[s3]; b_arr[s3] += c_chroma
        # Sector 4: R=X, B=C
        s4 = sector == 4; r_arr[s4] += x_val[s4]; b_arr[s4] += c_chroma
        # Sector 5: R=C, B=X
        s5 = sector == 5; r_arr[s5] += c_chroma; b_arr[s5] += x_val[s5]

        img_arr[py, px, 0] = (r_arr * 255).astype(np.uint8)
        img_arr[py, px, 1] = (g_arr * 255).astype(np.uint8)
        img_arr[py, px, 2] = (b_arr * 255).astype(np.uint8)

    return Image.fromarray(img_arr)


def render_color_image(
    grid_indices: np.ndarray, grid_mask: np.ndarray,
    label_to_rgb: dict[str, tuple[int, int, int]],
    bead_labels: list[str],
    cell_size: int = CELL_SIZE,
    transparent_labels: set[str] | None = None,
) -> Image.Image:
    """Render a plain bead color image — no grid lines, no labels, no legend."""
    if transparent_labels is None:
        transparent_labels = {"H1"}
    n_rows, n_cols = grid_mask.shape
    w = n_cols * cell_size
    h = n_rows * cell_size
    img = Image.new("RGB", (w, h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            if grid_mask[r, c]:
                label = bead_labels[grid_indices[r, c]]
                rgb = label_to_rgb[label]
                _draw_cell(draw, x0, y0, x1, y1, rgb,
                           label in transparent_labels, cell_size)
            else:
                _draw_checkerboard(draw, x0, y0, x1, y1, cell_size)
    return img


def render_output(
    grid_indices: np.ndarray, grid_mask: np.ndarray,
    bead_labels: list[str], label_to_rgb: dict[str, tuple[int, int, int]],
    cell_size: int = CELL_SIZE,
    origin: str = "bl",
    transparent_labels: set[str] | None = None,
    guide_lines: bool = False,
) -> Image.Image:
    """Render the bead pattern sheet with labels and legend.

    origin: "bl" (bottom-left), "tl" (top-left), "br" (bottom-right), "tr" (top-right)
    guide_lines: if True, every 5th grid line is darker, every 10th even darker.
    """
    from collections import Counter
    if transparent_labels is None:
        transparent_labels = {"H1"}

    n_rows, n_cols = grid_mask.shape

    # Origin determines numbering direction
    row_from_bottom = origin.startswith("b")
    col_from_right = origin.endswith("r")

    # Coordinate label margin (all 4 sides)
    coord_font_size = max(9, cell_size // 4)
    coord_font = _load_font(coord_font_size)
    max_label = str(max(n_rows, n_cols))
    bbox_test = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), max_label, font=coord_font)
    label_w = bbox_test[2] - bbox_test[0]
    label_h = bbox_test[3] - bbox_test[1]
    margin_lr = label_w + 8   # left and right
    margin_tb = label_h + 6   # top and bottom

    grid_w = n_cols * cell_size + GRID_LINE_WIDTH
    grid_h = n_rows * cell_size + GRID_LINE_WIDTH

    # Count color usage for legend
    usage: Counter[str] = Counter()
    for r in range(n_rows):
        for c in range(n_cols):
            if grid_mask[r, c]:
                usage[bead_labels[grid_indices[r, c]]] += 1

    # Legend layout
    legend_items = sorted(usage.items(), key=lambda x: x[0])
    swatch_size = cell_size
    legend_font_size = max(12, cell_size * 2 // 5)
    legend_font = _load_font(legend_font_size)
    legend_row_h = swatch_size + 8
    legend_item_w = swatch_size + legend_font_size * 7 + 16
    content_w = margin_lr * 2 + grid_w
    legend_cols = max(1, content_w // legend_item_w) if legend_items else 1
    legend_cols = min(legend_cols, len(legend_items)) if legend_items else 1
    legend_rows_count = math.ceil(len(legend_items) / legend_cols) if legend_items else 0
    legend_h = legend_rows_count * legend_row_h + legend_font_size + 24 if legend_items else 0

    total_w = max(content_w, legend_cols * legend_item_w)
    total_h = margin_tb * 2 + grid_h + legend_h

    img = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Grid area offset
    ox, oy = margin_lr, margin_tb

    # Font for bead labels
    font_size = max(12, int(cell_size * 0.48))
    font = _load_font(font_size)

    # Draw coordinate labels on all 4 sides
    coord_color = (100, 100, 100)

    def _col_label(c: int) -> str:
        return str(n_cols - c if col_from_right else c + 1)

    def _row_label(r: int) -> str:
        return str(n_rows - r if row_from_bottom else r + 1)

    for c in range(n_cols):
        lbl = _col_label(c)
        bbox = draw.textbbox((0, 0), lbl, font=coord_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = ox + c * cell_size + (cell_size - tw) // 2
        # Top
        draw.text((cx, (margin_tb - th) // 2), lbl, fill=coord_color, font=coord_font)
        # Bottom
        draw.text((cx, oy + grid_h + (margin_tb - th) // 2),
                  lbl, fill=coord_color, font=coord_font)

    for r in range(n_rows):
        lbl = _row_label(r)
        bbox = draw.textbbox((0, 0), lbl, font=coord_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cy = oy + r * cell_size + (cell_size - th) // 2
        # Left
        draw.text((margin_lr - tw - 4, cy), lbl, fill=coord_color, font=coord_font)
        # Right
        draw.text((ox + grid_w + 4, cy), lbl, fill=coord_color, font=coord_font)

    # Draw grid cells
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = ox + c * cell_size
            y0 = oy + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            if grid_mask[r, c]:
                idx = grid_indices[r, c]
                label = bead_labels[idx]
                rgb = label_to_rgb[label]
                _draw_cell(draw, x0, y0, x1, y1, rgb,
                           label in transparent_labels, cell_size)
                tc = _text_color(*rgb)
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x0 + (cell_size - tw) // 2
                ty = y0 + (cell_size - th) // 2
                draw.text((tx, ty), label, fill=tc, font=font)
            else:
                _draw_checkerboard(draw, x0, y0, x1, y1, cell_size)

    # Draw grid lines
    line_color = (180, 180, 180)
    line5_color = (130, 130, 130)   # every 5th line
    line10_color = (80, 80, 80)     # every 10th line
    for r in range(n_rows + 1):
        y = oy + r * cell_size
        rd = (n_rows - r) if row_from_bottom else r
        if guide_lines and rd % 10 == 0:
            draw.line([(ox, y), (ox + grid_w, y)], fill=line10_color, width=2)
        elif guide_lines and rd % 5 == 0:
            draw.line([(ox, y), (ox + grid_w, y)], fill=line5_color, width=2)
        else:
            draw.line([(ox, y), (ox + grid_w, y)], fill=line_color, width=GRID_LINE_WIDTH)
    for c in range(n_cols + 1):
        x = ox + c * cell_size
        cd = (n_cols - c) if col_from_right else c
        if guide_lines and cd % 10 == 0:
            draw.line([(x, oy), (x, oy + grid_h)], fill=line10_color, width=2)
        elif guide_lines and cd % 5 == 0:
            draw.line([(x, oy), (x, oy + grid_h)], fill=line5_color, width=2)
        else:
            draw.line([(x, oy), (x, oy + grid_h)], fill=line_color, width=GRID_LINE_WIDTH)

    # Draw legend
    if legend_items:
        legend_y_start = oy + grid_h + margin_tb + 10
        title_font = _load_cjk_font(legend_font_size + 2)
        total_beads = sum(count for _, count in legend_items)
        draw.text((8, legend_y_start),
                  f"颜色: {len(legend_items)} | 豆数: {total_beads} | 豆板: {n_cols}×{n_rows}",
                  fill=(0, 0, 0), font=title_font)
        legend_y_start += legend_font_size + 16

        for i, (label, count) in enumerate(legend_items):
            col = i % legend_cols
            row = i // legend_cols
            lx = col * legend_item_w + 8
            ly = legend_y_start + row * legend_row_h

            rgb = label_to_rgb[label]
            draw.rectangle([lx, ly, lx + swatch_size, ly + swatch_size],
                           fill=rgb, outline=(120, 120, 120))
            tc = _text_color(*rgb)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((lx + (swatch_size - tw) // 2, ly + (swatch_size - th) // 2),
                      label, fill=tc, font=font)
            draw.text((lx + swatch_size + 6, ly + (swatch_size - legend_font_size) // 2),
                      f"x{count}", fill=(0, 0, 0), font=legend_font)

    return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="拼豆图纸生成器 - Bead Pattern Generator"
    )
    parser.add_argument("input", help="Input pixel art image path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output image path (default: <input>_beadify.png)")
    parser.add_argument("-t", "--tolerance", type=float, default=None,
                        help="Color consolidation tolerance (default: 15 for cie76, 10 for ciede2000)")
    parser.add_argument("-e", "--extended", action="store_true",
                        help="Allow extended color series (P/Q/R/T/Y/ZG)")
    parser.add_argument("-c", "--cell-size", type=int, default=60,
                        help="Output cell size in pixels (default: 60)")
    parser.add_argument("-b", "--block-size", type=int, default=0,
                        help="Override auto-detected block size (0=auto)")
    parser.add_argument("--origin", choices=["bl", "tl", "br", "tr"], default="bl",
                        help="Coordinate origin: bl=bottom-left (default), tl=top-left, br=bottom-right, tr=top-right")
    parser.add_argument("--sample-ratio", type=float, default=0.5,
                        help="Fraction of each grid cell to sample for color (0.1-1.0, default: 0.5)")
    parser.add_argument("--metric", choices=["cie76", "ciede2000", "hybrid"], default="hybrid",
                        help="Color distance metric (default: hybrid)")
    args = parser.parse_args()

    if args.tolerance is None:
        args.tolerance = {"cie76": 15.0, "ciede2000": 10.0, "hybrid": 12.0}.get(
            args.metric, 12.0)

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_beadify.png"
    else:
        output_path = Path(args.output)

    # Locate color data next to this script
    json_path = Path(__file__).parent / "colors" / "mard.json"
    if not json_path.exists():
        # Fallback to legacy location
        json_path = Path(__file__).parent / "bead_colors.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return

    # 1. Load bead colors
    print(f"Loading bead colors (extended={args.extended})...")
    bead_lab, bead_labels, label_to_rgb = load_bead_colors(
        str(json_path), args.extended
    )
    print(f"  {len(bead_labels)} bead colors available")

    # 2. Load image
    print(f"Loading image: {input_path}")
    img_pil = Image.open(input_path).convert("RGBA")
    img = np.array(img_pil)
    print(f"  Image size: {img.shape[1]}x{img.shape[0]}")

    # 3. Detect grid
    print("Detecting grid...")
    block_size = detect_block_size(img, override=args.block_size)
    print(f"  Block size: {block_size}")
    row_bounds, col_bounds = detect_grid(img, block_size)
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    print(f"  Grid: {n_cols}x{n_rows} blocks")

    # 4. Extract block colors
    print("Extracting block colors...")
    grid_lab, grid_mask = extract_block_colors(img, row_bounds, col_bounds,
                                                sample_ratio=args.sample_ratio)
    opaque_count = int(grid_mask.sum())
    print(f"  {opaque_count} opaque blocks")

    # 5. Map to bead colors
    print("Mapping to bead colors...")
    grid_indices, grid_lab = map_to_bead_colors(
        grid_lab, grid_mask, bead_lab, bead_labels, metric=args.metric
    )
    unique_before = len(set(grid_indices[grid_mask].tolist()))
    print(f"  {unique_before} unique colors before consolidation")

    # 6. Consolidate
    if args.tolerance > 0:
        print(f"Consolidating colors (tolerance={args.tolerance})...")
        grid_indices = consolidate_colors(
            grid_indices, grid_mask, grid_lab, bead_lab,
            bead_labels, args.tolerance, metric=args.metric
        )
    unique_after = len(set(grid_indices[grid_mask].tolist()))
    print(f"  {unique_after} unique colors after consolidation")

    # 7. Render output
    print("Rendering output...")
    result = render_output(
        grid_indices, grid_mask, bead_labels, label_to_rgb, args.cell_size,
        origin=args.origin,
    )
    result.save(str(output_path))
    print(f"Saved: {output_path}")

    # Summary
    from collections import Counter
    usage: Counter[str] = Counter()
    for r in range(grid_mask.shape[0]):
        for c in range(grid_mask.shape[1]):
            if grid_mask[r, c]:
                usage[bead_labels[grid_indices[r, c]]] += 1
    print(f"\nColor usage ({len(usage)} colors, {opaque_count} beads total):")
    for label, count in sorted(usage.items(), key=lambda x: -x[1]):
        rgb = label_to_rgb[label]
        print(f"  {label:>4s}: {count:4d}  #{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")


if __name__ == "__main__":
    main()
