from __future__ import annotations

import copy
import csv
import json
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter, gaussian_filter1d, grey_opening
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

# ---- optional GPU ---------------------------------------------------------
try:  # pragma: no cover
    import cupy as _cp
    from cupyx.scipy.ndimage import gaussian_filter as _cp_gauss

    GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    _cp = None
    GPU = False


def gblur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, on the GPU when CuPy is available."""
    if sigma <= 0:
        return a
    if GPU and a.size > 250_000:
        return _cp.asnumpy(_cp_gauss(_cp.asarray(a), sigma))
    return gaussian_filter(a, sigma)


# The preview canvas now fills the window; these are only fall-backs used
# before the window has been laid out for the first time.
MAX_PREVIEW_W = 1600
MAX_PREVIEW_H = 900
ZOOM_STEPS = (0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
HANDLE = 6  # half-size of a resize handle, in canvas pixels

# Each blot row gets its own colour motif (row 1 green, row 2 blue, row 3 red...)
ROW_COLORS = ("#00b050", "#2979ff", "#ff3d3d", "#ff9800", "#ab47bc", "#00bcd4")
ROW_TINTS = ("#E7F5EC", "#E6EFFB", "#FBE9E9", "#FDF0E1", "#F2E9F7", "#E3F5F7")


def row_color(row: int) -> str:
    return ROW_COLORS[(max(1, int(row)) - 1) % len(ROW_COLORS)]


def row_tint(row: int) -> str:
    return ROW_TINTS[(max(1, int(row)) - 1) % len(ROW_TINTS)]


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class Band:
    row: int               # which horizontal blot row (1-based)
    lane: int              # lane number inside that row
    index: int             # band index inside the lane
    x: int
    y: int
    w: int
    h: int
    raw_volume: float
    bg_volume: float
    net_volume: float
    peak: float
    centre_y: float
    sigma: float
    snr: float
    empty: bool = False
    manual: bool = False   # created or edited by hand
    auc: float = 0.0       # area under the 1-D lane intensity "hill", matching
                            # the shaded region shown in the profile popup

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.row, self.lane, self.index)


@dataclass
class Result:
    bands: list = field(default_factory=list)
    density: np.ndarray | None = None
    gray: np.ndarray | None = None
    angle: float = 0.0


# --------------------------------------------------------------------------
# 1. density space
# --------------------------------------------------------------------------
def to_density(gray: np.ndarray, dark_bands: bool = True) -> np.ndarray:
    img = gray.astype(np.float32)
    if dark_bands:
        white = float(np.percentile(img, 97))
        img = np.clip(white - img, 0, None)
    else:
        black = float(np.percentile(img, 3))
        img = np.clip(img - black, 0, None)
    return img


# --------------------------------------------------------------------------
# 2. rolling-ball background
# --------------------------------------------------------------------------
def rolling_ball(sig: np.ndarray, radius: int = 30, downscale: int = 4) -> np.ndarray:
    r = max(4, int(radius))
    small = cv2.resize(sig, None, fx=1.0 / downscale, fy=1.0 / downscale,
                       interpolation=cv2.INTER_AREA)
    rs = max(2, r // downscale)
    yy, xx = np.mgrid[-rs:rs + 1, -rs:rs + 1]
    d2 = (xx ** 2 + yy ** 2) / float(rs ** 2)
    inside = d2 <= 1.0
    height = np.zeros_like(d2, dtype=np.float32)
    height[inside] = (1.0 - d2[inside]) * rs
    bg_small = grey_opening(small, structure=height, mode="nearest")
    bg_small = gaussian_filter(bg_small, rs / 2.0)
    bg = cv2.resize(bg_small, (sig.shape[1], sig.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
    return np.clip(sig - bg, 0, None)


# --------------------------------------------------------------------------
# 3. deskew
# --------------------------------------------------------------------------
def _profile_sharpness(sig: np.ndarray) -> float:
    prof = sig.mean(axis=1)
    prof = prof - prof.min()
    if prof.max() <= 0:
        return 0.0
    prof = prof / prof.max()
    return float(np.sum(np.diff(prof) ** 2))


def estimate_skew(sig: np.ndarray, max_deg: float = 6.0) -> float:
    thr = np.percentile(sig, 99.0)
    if thr <= 0:
        return 0.0
    work = cv2.resize(sig, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    best, best_a = -1.0, 0.0
    for step, span, centre in ((1.0, max_deg, 0.0), (0.2, 1.0, None), (0.05, 0.2, None)):
        c = best_a if centre is None else centre
        for a in np.arange(c - span, c + span + 1e-9, step):
            s = _profile_sharpness(rotate(work, float(a)))
            if s > best:
                best, best_a = s, float(a)
    return best_a


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# 4. blot rows
# --------------------------------------------------------------------------
def find_rows(sig: np.ndarray, min_h: int = 6) -> list[tuple[int, int]]:
    h = sig.shape[0]
    prof = gaussian_filter1d(sig.mean(axis=1), max(1.0, h / 200.0))
    if prof.max() <= 0:
        return [(0, h)]
    noise = 1.4826 * np.median(np.abs(prof - np.median(prof))) + 1e-6
    peaks, _ = find_peaks(prof, prominence=max(3 * noise, 0.08 * prof.max()),
                          distance=max(min_h, h // 40))
    if len(peaks) == 0:
        return [(0, h)]
    widths = peak_widths(prof, peaks, rel_height=0.9)
    rows = []
    for i, pk in enumerate(peaks):
        lo = int(max(0, np.floor(widths[2][i]) - 3))
        hi = int(min(h, np.ceil(widths[3][i]) + 3))
        if hi - lo >= min_h:
            rows.append((lo, hi))
    if not rows:
        return [(0, h)]
    rows.sort()
    gap_tol = max(min_h * 2, 12)
    merged = [rows[0]]
    for lo, hi in rows[1:]:
        if lo <= merged[-1][1] + gap_tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


# --------------------------------------------------------------------------
# 5. lanes
# --------------------------------------------------------------------------
def _pitch_from_peaks(peaks: np.ndarray) -> float:
    if len(peaks) < 3:
        return 0.0
    d = np.diff(np.sort(peaks)).astype(float)
    med = float(np.median(d))
    keep = d[(d > 0.6 * med) & (d < 1.6 * med)]
    return float(np.mean(keep)) if keep.size else med


def find_lanes(sig: np.ndarray, band: tuple[int, int], sensitivity: float,
               min_lane_w: int = 6, fill_grid: bool = True):
    y0, y1 = band
    strip = sig[y0:y1, :]
    w = strip.shape[1]
    prof = gaussian_filter1d(strip.sum(axis=0), max(1.0, w / 400.0))
    if prof.max() <= 0:
        return np.array([], dtype=int), 0.0, prof
    trend = gaussian_filter1d(prof, max(6.0, w / 12.0))
    det = prof - trend
    noise = 1.4826 * np.median(np.abs(det - np.median(det))) + 1e-6
    prom = max(1.0 * noise, 0.04 * (det.max() - det.min())) / max(sensitivity, 0.2)
    peaks, _ = find_peaks(det, prominence=prom, distance=min_lane_w)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(det))])

    pitch = _pitch_from_peaks(peaks)

    # signal floor a candidate lane centre must clear to be considered real
    occupied_thr = max(0.8 * noise, 0.008 * (det.max() - det.min()))

    if pitch >= min_lane_w and len(peaks) >= 2:
        # Merge two peaks only when they are close AND the dip between them is
        # shallow.  Distance alone used to fuse two genuinely separate
        # neighbouring lanes (typically at the outer edges) into one wide box.
        strengths = det[peaks]
        merged = [0]
        for i in range(1, len(peaks)):
            a, b = int(peaks[merged[-1]]), int(peaks[i])
            valley = float(det[a:b + 1].min()) if b > a else float(det[a])
            weaker = min(float(det[a]), float(det[b]))
            depth = weaker - valley
            separate = depth > max(0.25 * max(weaker - float(det.min()), 0.0),
                                   2.0 * noise)
            if (b - a) < 0.45 * pitch and not separate:
                if strengths[i] > strengths[merged[-1]]:
                    merged[-1] = i
            else:
                merged.append(i)
        peaks = peaks[merged]

    if fill_grid and pitch >= min_lane_w and len(peaks) >= 2:
        centres = [int(peaks[0])]
        for a, b in zip(peaks[:-1], peaks[1:]):
            n = max(1, int(round((b - a) / pitch)))
            for k in range(1, n):
                c = a + (b - a) * k / n
                lo = int(max(a + 2, c - 0.28 * pitch))
                hi = int(min(b - 2, c + 0.28 * pitch))
                # keep interior grid lanes even when faint: whether a band is
                # really there is decided later, from the lane profile
                c = lo + int(np.argmax(det[lo:hi])) if hi > lo else c
                centres.append(int(round(c)))
            centres.append(int(b))

        edge_thr = occupied_thr
        c = centres[0] - pitch
        while c - 0.4 * pitch >= 0:
            lo = int(max(0, c - 0.28 * pitch))
            hi = int(min(w, c + 0.28 * pitch))
            if hi <= lo or det[lo:hi].max() < edge_thr:
                break
            centres.insert(0, int(round(lo + np.argmax(det[lo:hi]))))
            c = centres[0] - pitch
        c = centres[-1] + pitch
        while c + 0.4 * pitch < w:
            lo = int(max(0, c - 0.28 * pitch))
            hi = int(min(w, c + 0.28 * pitch))
            if hi <= lo or det[lo:hi].max() < edge_thr:
                break
            centres.append(int(round(lo + np.argmax(det[lo:hi]))))
            c = centres[-1] + pitch
        peaks = np.array(sorted(centres))

    return np.array(sorted(set(int(p) for p in peaks))), float(pitch), prof


def consensus_centres(per_row: list[np.ndarray], pitch: float,
                      tol_frac: float = 0.45) -> np.ndarray:
    arrays = [np.asarray(a, dtype=float) for a in per_row if len(a)]
    if not arrays:
        return np.array([], dtype=int)
    master = max(arrays, key=len).copy()
    tol = max(3.0, tol_frac * (pitch or 10.0))
    for arr in arrays:
        if arr is master or len(arr) == 0:
            continue
        cand = [0.0] + [m - a for m in master for a in arr if abs(m - a) < tol]
        best, best_score = 0.0, -1
        for s in cand:
            score = sum(1 for a in arr if np.min(np.abs(master - (a + s))) < tol)
            if score > best_score:
                best, best_score = s, score
        for a in arr + best:
            if np.min(np.abs(master - a)) > tol:
                master = np.append(master, a)
        master = np.sort(master)
    return np.array([int(round(v)) for v in np.sort(master)])


def lane_boxes(centres: np.ndarray, prof: np.ndarray, pitch: float,
               w: int, min_lane_w: int = 6):
    half = max(min_lane_w // 2, int(round((pitch or min_lane_w) * 0.42)))
    lanes = []
    for i, pk in enumerate(centres):
        pk = int(pk)
        if i == 0:
            left = max(0, pk - half)
        else:
            lo, hi = int(centres[i - 1]), pk
            left = lo + int(np.argmin(prof[lo:hi])) if hi > lo + 1 else pk - half
        if i == len(centres) - 1:
            right = min(w, pk + half)
        else:
            lo, hi = pk, int(centres[i + 1])
            right = lo + int(np.argmin(prof[lo:hi])) if hi > lo + 1 else pk + half
        if pitch:
            left = max(left, int(pk - pitch * 0.5))
            right = min(right, int(pk + pitch * 0.5))
        left, right = max(0, left), min(w, right)
        if right - left >= max(3, min_lane_w // 2):
            lanes.append((int(left), int(right)))
    return lanes


# --------------------------------------------------------------------------
# 6. bands inside a lane
# --------------------------------------------------------------------------
def _gauss(x, a, mu, sd, c):
    return a * np.exp(-0.5 * ((x - mu) / sd) ** 2) + c


def _edge_strictness(sig_shape, y_abs: float, x0: int, x1: int) -> float:
    """1.0 in the middle of the blot, mildly higher for bands sitting on the
    very edge of the image, where vignetting / cropping artefacts can make a
    single band look like two.  Kept modest (max 1.45): a strong boost here
    used to fuse two real outer bands into one big box."""
    h, w = sig_shape[:2]
    my = max(6.0, 0.06 * h)
    mx = max(6.0, 0.06 * w)
    dy = min(float(y_abs), float(h - 1 - y_abs))
    dx = min(float(x0), float(w - x1))
    t = max(0.0, 1.0 - min(dy / my, 1.0)) * 0.3 + max(0.0, 1.0 - min(dx / mx, 1.0)) * 0.2
    return float(1.0 + min(t, 0.45))


def _merge_split_peaks(prof: np.ndarray, peaks: np.ndarray, base: float,
                       noise: float, strictness: float,
                       min_band_h: int) -> np.ndarray:
    """Collapse peak pairs that are really ONE band: a shallow valley between
    them, or centres closer than a plausible band height."""
    if len(peaks) < 2:
        return peaks
    peaks = np.sort(np.asarray(peaks, dtype=int))
    frac = min(0.60, 0.28 * strictness)          # required valley depth
    min_sep = max(min_band_h * 2, int(round(3 * strictness)))
    changed = True
    while changed and len(peaks) > 1:
        changed = False
        for i in range(len(peaks) - 1):
            a, b = int(peaks[i]), int(peaks[i + 1])
            pa, pb = float(prof[a]), float(prof[b])
            valley = float(prof[a:b + 1].min())
            weaker = min(pa, pb) - base
            depth = min(pa, pb) - valley
            need = max(frac * max(weaker, 0.0), 2.0 * noise * strictness)
            if depth < need or (b - a) < min_sep:
                keep = a if pa >= pb else b
                peaks = np.delete(peaks, i + 1 if keep == a else i)
                changed = True
                break
    return peaks


def find_bands_in_lane(sig: np.ndarray, x0: int, x1: int, y0: int, y1: int,
                       sensitivity: float, min_band_h: int = 3):
    pad = max(4, (y1 - y0) // 2)
    a, b = max(0, int(y0) - pad), min(sig.shape[0], int(y1) + pad)
    prof = gaussian_filter1d(sig[a:b, x0:x1].mean(axis=1), 1.2)
    if prof.max() <= 0:
        return []
    base = float(np.percentile(prof, 20))
    noise = 1.4826 * np.median(np.abs(prof - np.median(prof))) + 1e-6
    strictness = _edge_strictness(sig.shape, (y0 + y1) / 2.0, int(x0), int(x1))
    prom = max(1.6 * noise, 0.06 * (prof.max() - base)) / max(sensitivity, 0.2)
    prom *= 0.5 * (1.0 + strictness)   # harder to split near the image edge
    peaks, _ = find_peaks(prof, prominence=prom, distance=max(min_band_h,
                                        int(round(min_band_h * strictness))))
    peaks = np.array([p for p in peaks if y0 <= a + p < y1], dtype=int)
    if len(peaks) == 0:
        # Faint / smeared band: no clear prominence peak, but the lane still
        # carries signal.  Take its strongest point instead of dropping the
        # whole lane (this is what made entire bands disappear).
        inner = prof[max(0, int(y0) - a):max(1, int(y1) - a)]
        if inner.size and (float(inner.max()) - base) > 1.5 * noise:
            peaks = np.array([int(np.argmax(inner)) + max(0, int(y0) - a)],
                             dtype=int)
        else:
            return []
    peaks = _merge_split_peaks(prof, peaks, base, noise, strictness, min_band_h)

    widths = peak_widths(prof, peaks, rel_height=0.85)
    out = []
    for i, pk in enumerate(peaks):
        lo = float(widths[2][i])
        hi = float(widths[3][i])
        mu, sd = float(pk), max((hi - lo) / 2.355, 1.0)
        s0 = int(max(0, pk - 3 * sd)), int(min(len(prof), pk + 3 * sd + 1))
        xs = np.arange(*s0, dtype=float)
        if xs.size >= 5:
            try:
                p, _ = curve_fit(_gauss, xs, prof[s0[0]:s0[1]],
                                 p0=[prof[pk] - base, mu, sd, base], maxfev=4000)
                if 0 < p[2] < (b - a) and abs(p[1] - pk) < 3 * sd:
                    mu, sd = float(p[1]), float(abs(p[2]))
            except Exception:
                pass
        top = mu - 3.0 * sd
        bot = mu + 3.0 * sd
        cut = base + 0.02 * (prof[pk] - base)
        j = int(round(mu))
        while top < j and prof[max(0, int(top))] < cut:
            top += 1
        while bot > j and prof[min(len(prof) - 1, int(bot))] < cut:
            bot -= 1
        t = max(int(y0), int(round(a + top)))
        bt = min(int(y1), int(round(a + bot)) + 1)
        if bt - t >= min_band_h:
            out.append((t, bt, a + mu, sd, float((prof[pk] - base) / noise)))

    # --- final safety net: fuse segments whose boxes still overlap a lot -----
    out.sort(key=lambda s: s[0])
    fused = []
    for seg in out:
        if fused:
            pt, pb_, pcy, psd, psnr = fused[-1]
            t, bt, cy, sd, snr = seg
            ov = min(pb_, bt) - max(pt, t)
            if ov > 0 and ov > 0.45 * min(pb_ - pt, bt - t):
                keep_cy, keep_sd = (pcy, psd) if psnr >= snr else (cy, sd)
                fused[-1] = (min(pt, t), max(pb_, bt), keep_cy, keep_sd,
                             max(psnr, snr))
                continue
        fused.append(seg)
    return fused


# --------------------------------------------------------------------------
# 6b. tight ROI fitting  (shrink-wrap the box onto the real band footprint)
# --------------------------------------------------------------------------
def _extent_from_profile(prof: np.ndarray, centre: int, frac: float = 0.02,
                         coverage: float = 0.99, noise_k: float = 1.0
                         ) -> tuple[int, int]:
    """Extent of a band around `centre` that captures essentially ALL of its
    signal.

    Two criteria are combined and the WIDER of the two wins:

      1. Threshold criterion - grow while the profile stays above
         base + max(frac * amp, noise_k * sigma_noise).  `frac` is deliberately
         low (2%) so diffuse band tails are not clipped.
      2. Coverage criterion - keep growing outward, always taking the side with
         the larger remaining signal, until `coverage` (default 99%) of the
         above-baseline area of the peak is enclosed, or until the profile has
         decayed into the noise floor on both sides.

    Growth also refuses to stop while the profile is still descending steeply,
    which is what previously truncated smeared / saturated bands.
    Returns (lo, hi) as an inclusive/exclusive pair.
    """
    if prof.size == 0:
        return 0, 1
    n = int(prof.size)
    centre = int(np.clip(centre, 0, n - 1))
    base = float(np.percentile(prof, 10))
    amp = float(prof.max() - base)
    if amp <= 0:
        return 0, n

    resid = prof - base
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-6
    thr = max(frac * amp, noise_k * noise)

    # --- 1. threshold run -------------------------------------------------
    lo = centre
    while lo > 0 and (prof[lo - 1] - base) > thr:
        lo -= 1
    hi = centre
    while hi < n - 1 and (prof[hi + 1] - base) > thr:
        hi += 1

    # --- 2. coverage growth ----------------------------------------------
    pos = np.clip(resid, 0.0, None)
    # total area of the peak's own basin (walk out to the local minima)
    bl = lo
    while bl > 0 and pos[bl - 1] <= pos[bl] + 1e-9:
        bl -= 1
    bh = hi
    while bh < n - 1 and pos[bh + 1] <= pos[bh] + 1e-9:
        bh += 1
    total = float(pos[bl:bh + 1].sum())
    if total > 0:
        inside = float(pos[lo:hi + 1].sum())
        floor = 0.35 * noise  # stop once both flanks are pure background
        while inside < coverage * total and (lo > bl or hi < bh):
            left = pos[lo - 1] if lo > bl else -1.0
            right = pos[hi + 1] if hi < bh else -1.0
            if max(left, right) <= floor:
                break
            if left >= right:
                lo -= 1
                inside += float(left)
            else:
                hi += 1
                inside += float(right)

    return lo, hi + 1



# --------------------------------------------------------------------------
# 6b. LEARNABLE ROI MODEL  (new in v3.3)
# --------------------------------------------------------------------------
# The ROI fitter is driven by five continuous parameters.  Every time you
# correct the boxes by hand you can store that blot as a training example; the
# trainer then searches the parameter space so the automatic fit reproduces
# your own boxes as closely as possible (maximum mean IoU).  Everything is
# numpy-only, so no extra dependencies are needed.

BQ_DIR = Path.home() / ".wbdetect"
TRAIN_DIR = BQ_DIR / "roi_training"
MODEL_PATH = BQ_DIR / "roi_model.json"

DEFAULT_ROI_PARAMS = {
    "frac": 0.02,        # profile threshold as a fraction of band peak
    "coverage": 0.99,    # fraction of the band signal that must be inside
    "pad": 2,            # minimum pad in px
    "pad_frac": 0.10,    # pad as a fraction of the fitted size
    "edge_frac": 0.05,   # border signal allowed before the box grows
}

ROI_PARAM_GRID = {
    "frac": [0.003, 0.006, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12],
    "coverage": [0.85, 0.90, 0.94, 0.97, 0.99, 0.995, 0.999],
    "pad": [0, 1, 2, 3, 5],
    "pad_frac": [0.0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35],
    "edge_frac": [0.02, 0.05, 0.10, 0.20, 0.35, 0.60],
}


class RoiModel:
    """Trainable parameter set for the ROI fitter."""

    def __init__(self, params=None, enabled=False, samples=0, boxes=0,
                 iou=0.0, base_iou=0.0, trained=""):
        self.params = dict(DEFAULT_ROI_PARAMS)
        if params:
            self.params.update({k: params[k] for k in DEFAULT_ROI_PARAMS if k in params})
        self.enabled = bool(enabled)
        self.samples = int(samples)
        self.boxes = int(boxes)
        self.iou = float(iou)
        self.base_iou = float(base_iou)
        self.trained = trained

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls) -> "RoiModel":
        try:
            d = json.loads(MODEL_PATH.read_text())
            return cls(**d)
        except Exception:
            return cls()

    def save(self):
        BQ_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps({
            "params": self.params, "enabled": self.enabled,
            "samples": self.samples, "boxes": self.boxes,
            "iou": self.iou, "base_iou": self.base_iou,
            "trained": self.trained,
        }, indent=2))

    def reset(self):
        self.__init__()
        try:
            MODEL_PATH.unlink()
        except Exception:
            pass

    # -- use ---------------------------------------------------------------
    def active(self) -> dict:
        return dict(self.params) if self.enabled else dict(DEFAULT_ROI_PARAMS)

    def summary(self) -> str:
        if not self.samples:
            return "AI model: not trained yet (using built-in fit)"
        state = "ON" if self.enabled else "off"
        return (f"AI model [{state}]: {self.samples} blot(s), {self.boxes} ROIs, "
                f"match {self.iou * 100:.1f}% (built-in {self.base_iou * 100:.1f}%)")


ROI_MODEL = RoiModel.load()


# ---- training data --------------------------------------------------------
def save_training_sample(gray: np.ndarray, boxes: list[tuple[int, int, int, int]],
                         bg_radius: int, dark_bands: bool,
                         bg_regions: list[tuple[int, int, int, int]] | None = None) -> Path:
    """Store a corrected blot (image, your final boxes and the areas you told
    the AI to ignore) as a training example."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAIN_DIR / f"sample_{int(time.time() * 1000)}.npz"
    np.savez_compressed(path,
                        gray=gray.astype(np.uint16),
                        boxes=np.asarray(boxes, dtype=np.int32).reshape(-1, 4),
                        bg=np.asarray(bg_regions or [], dtype=np.int32).reshape(-1, 4),
                        bg_radius=np.int32(bg_radius),
                        dark=np.int32(bool(dark_bands)))
    return path


def list_training_samples() -> list[Path]:
    if not TRAIN_DIR.exists():
        return []
    return sorted(TRAIN_DIR.glob("sample_*.npz"))


def clear_training_samples():
    for p in list_training_samples():
        try:
            p.unlink()
        except Exception:
            pass


def _prep_signal(gray: np.ndarray, bg_radius: int, dark: bool) -> np.ndarray:
    sig = to_density(gray, dark)
    sig = gblur(sig, 0.8)
    return rolling_ball(sig, bg_radius)


def _iou(a, b) -> float:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _seed_box(box, shrink: float = 0.45):
    x0, x1, y0, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw = max(2.0, (x1 - x0) * shrink / 2.0)
    hh = max(1.5, (y1 - y0) * shrink / 2.0)
    return int(cx - hw), int(cx + hw), int(cy - hh), int(cy + hh)


def _score_params(data, params) -> float:
    """Mean IoU between the fitted boxes and the user's boxes."""
    scores = []
    for sig, boxes in data:
        h, w = sig.shape[:2]
        for box in boxes:
            sx0, sx1, sy0, sy1 = _seed_box(box)
            mx = max(6, int((box[1] - box[0]) * 1.2))
            my = max(6, int((box[3] - box[2]) * 1.2))
            xlim = (max(0, box[0] - mx), min(w, box[1] + mx))
            ylim = (max(0, box[2] - my), min(h, box[3] + my))
            try:
                fit = tighten_box(sig, sx0, sx1, sy0, sy1,
                                  xlim=xlim, ylim=ylim, **params)
            except Exception:
                continue
            scores.append(_iou(fit, box))
    return float(np.mean(scores)) if scores else 0.0


def train_roi_model(model: RoiModel = None, progress=None) -> tuple[bool, str]:
    """Coordinate-descent search over the ROI parameters on the saved blots."""
    model = model or ROI_MODEL
    samples = list_training_samples()
    if not samples:
        return False, "No training examples saved yet."

    data = []
    nboxes = 0
    for i, p in enumerate(samples, 1):
        if progress:
            progress(f"Preparing example {i}/{len(samples)} ...")
        try:
            z = np.load(p)
            boxes = [tuple(int(v) for v in b) for b in z["boxes"]
                     if b[1] > b[0] + 1 and b[3] > b[2] + 1]
            if not boxes:
                continue
            sig = _prep_signal(z["gray"].astype(np.float32),
                               int(z["bg_radius"]), bool(int(z["dark"])))
            data.append((sig, boxes))
            nboxes += len(boxes)
        except Exception:
            continue
    if not data:
        return False, "Training examples could not be read."

    base = _score_params(data, dict(DEFAULT_ROI_PARAMS))
    best = dict(model.params) if model.samples else dict(DEFAULT_ROI_PARAMS)
    best_score = _score_params(data, best)
    if base > best_score:
        best, best_score = dict(DEFAULT_ROI_PARAMS), base

    for sweep in range(3):
        improved = False
        for name, values in ROI_PARAM_GRID.items():
            if progress:
                progress(f"Training pass {sweep + 1}/3 - tuning {name} ...")
            for v in values:
                if v == best[name]:
                    continue
                trial = dict(best)
                trial[name] = v
                s = _score_params(data, trial)
                if s > best_score + 1e-5:
                    best, best_score, improved = trial, s, True
        if not improved:
            break

    model.params = best
    model.samples = len(data)
    model.boxes = nboxes
    model.iou = best_score
    model.base_iou = base
    model.enabled = True
    model.trained = time.strftime("%Y-%m-%d %H:%M")
    model.save()
    return True, (f"Trained on {len(data)} blot(s) / {nboxes} ROIs - "
                  f"box match improved from {base * 100:.1f}% to {best_score * 100:.1f}%.")



# --------------------------------------------------------------------------
# 6c. DEEP BAND NET  (new in v3.8)
# --------------------------------------------------------------------------
# A large, fully configurable neural network. It learns from the ROIs you
# accepted (positives), from everything else in the blot (negatives) and,
# most importantly, from the areas you explicitly painted as
# BACKGROUND / IGNORE.  Once trained it can throw away any automatically
# detected ROI that looks like background to it.
#
# The math (forward pass, Adam updates) runs through a small array-library
# indirection (`self._array_module()`): it transparently dispatches to CuPy
# on the GPU when a CUDA device is available and `use_gpu` is enabled in the
# config, and falls back to plain NumPy on the CPU otherwise. Only host-side
# glue (image cropping with cv2, file I/O) stays on NumPy/CPU, since cv2 and
# npz files don't understand device arrays.
#
#   layers  -> how many hidden layers (depth)
#   width   -> neurons per hidden layer
#   epochs  -> how many passes over the training patches
#   bits    -> numeric precision of the weights (16 / 32 / 64 bit floats)
#   use_gpu -> dispatch the math to CUDA via CuPy when a GPU is available

BANDNET_PATH = BQ_DIR / "bandnet.npz"
BANDNET_META = BQ_DIR / "bandnet.json"

DEEP_DEFAULTS = {
    "patch": 24,        # every ROI is resampled to patch x patch pixels
    "layers": 8,        # hidden layers
    "width": 256,       # neurons per hidden layer
    "epochs": 300,
    "bits": 32,
    "lr": 0.0015,
    "batch": 64,
    "threshold": 0.5,
    "use_gpu": True,    # use CUDA (via CuPy) for training/inference when available
}


def _dtype_for_bits(bits: int):
    return {16: np.float16, 32: np.float32, 64: np.float64}.get(int(bits), np.float32)


def _patch_vector(sig: np.ndarray, box, size: int) -> np.ndarray | None:
    """Crop a box (with a little context), resample it and normalise it."""
    x0, x1, y0, y1 = [int(v) for v in box]
    h, w = sig.shape[:2]
    mx = max(2, int((x1 - x0) * 0.25))
    my = max(2, int((y1 - y0) * 0.25))
    x0, x1 = max(0, x0 - mx), min(w, x1 + mx)
    y0, y1 = max(0, y0 - my), min(h, y1 + my)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = np.asarray(sig[y0:y1, x0:x1], dtype=np.float32)
    try:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        return None
    lo, hi = float(crop.min()), float(crop.max())
    norm = (crop - lo) / (hi - lo) if hi > lo else np.zeros_like(crop)
    # a couple of shape/contrast descriptors help a lot on flat background
    extra = np.array([
        min(1.0, (hi - lo) / 255.0),
        min(1.0, float(crop.mean()) / 255.0),
        min(1.0, float(crop.std()) / 64.0),
        min(1.0, (x1 - x0) / 200.0),
        min(1.0, (y1 - y0) / 200.0),
    ], dtype=np.float32)
    return np.concatenate([norm.ravel(), extra])


class DeepBandNet:
    """Configurable deep MLP classifier: band (1) vs background (0).

    All heavy math is routed through ``self._array_module()``, which returns
    the CuPy module when CUDA is available and ``cfg["use_gpu"]`` is true, or
    NumPy otherwise. Weights/moments/activations therefore live on whichever
    device is active during training or inference; they are only ever moved
    back to host NumPy arrays at the CPU/GPU boundary (`predict`'s return
    value, and `save()`'s npz payload).
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(DEEP_DEFAULTS)
        if cfg:
            self.cfg.update({k: cfg[k] for k in DEEP_DEFAULTS if k in cfg})
        self.W: list = []
        self.b: list = []
        self.enabled = False
        self.trained = ""
        self.samples = 0
        self.accuracy = 0.0
        self.params = 0

    # -- backend -------------------------------------------------------
    def gpu_active(self) -> bool:
        """True if this net will actually dispatch its math to CUDA."""
        return bool(GPU and _cp is not None and self.cfg.get("use_gpu", True))

    def _array_module(self):
        return _cp if self.gpu_active() else np

    @staticmethod
    def _to_host(arr) -> np.ndarray:
        """Bring a (possibly CuPy) array back to host NumPy memory."""
        if _cp is not None and isinstance(arr, _cp.ndarray):
            return _cp.asnumpy(arr)
        return np.asarray(arr)

    def _to_float(self, val) -> float:
        """Safely convert a 0-d NumPy/CuPy array (or scalar) to a Python float."""
        if _cp is not None and isinstance(val, _cp.ndarray):
            return float(_cp.asnumpy(val))
        return float(val)

    # -- architecture ------------------------------------------------------
    def _sizes(self, n_in: int) -> list[int]:
        return ([n_in] + [int(self.cfg["width"])] * int(self.cfg["layers"]) + [1])

    def _init(self, n_in: int, seed: int = 0):
        # Weights are always drawn with NumPy's RNG (so initialization is
        # identical and reproducible whether or not a GPU is used), then
        # moved onto whichever backend is active.
        xp = self._array_module()
        rng = np.random.default_rng(seed)
        dt = _dtype_for_bits(self.cfg["bits"])
        sizes = self._sizes(n_in)
        self.W, self.b = [], []
        for a, c in zip(sizes[:-1], sizes[1:]):
            w = rng.normal(0, np.sqrt(2.0 / a), (a, c)).astype(dt)
            bch = np.zeros(c, dtype=dt)
            self.W.append(xp.asarray(w))
            self.b.append(xp.asarray(bch))
        self.params = int(sum(int(w.size) for w in self.W) + sum(int(v.size) for v in self.b))

    # -- forward / backward ------------------------------------------------
    def _forward(self, X):
        xp = self._array_module()
        acts = [X]
        a = X
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            a = z if i == len(self.W) - 1 else xp.maximum(z, 0)
            acts.append(a)
        # sigmoid on the (float32-promoted) logit for numerical safety
        logit = acts[-1].astype(xp.float32)
        logit = xp.clip(logit, -30, 30)
        p = 1.0 / (1.0 + xp.exp(-logit))
        return acts, p

    def _predict_xp(self, X):
        """Forward pass that stays on-device (no host round-trip); used
        internally for fast accuracy checks during training."""
        _, p = self._forward(X)
        return p.ravel()

    def predict(self, X) -> np.ndarray:
        if not self.W:
            return np.zeros(len(X), dtype=np.float32)
        xp = self._array_module()
        dt = _dtype_for_bits(self.cfg["bits"])
        Xd = xp.asarray(np.asarray(X, dtype=dt))
        p = self._predict_xp(Xd)
        return self._to_host(p).astype(np.float32)

    def fit(self, X, y, progress=None, should_stop=None) -> float:
        xp = self._array_module()
        dt = _dtype_for_bits(self.cfg["bits"])
        X = xp.asarray(np.asarray(X, dtype=dt))
        y = xp.asarray(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        self._init(X.shape[1])
        lr = float(self.cfg["lr"])
        batch = max(8, int(self.cfg["batch"]))
        epochs = max(1, int(self.cfg["epochs"]))
        rng = np.random.default_rng(1)  # permutation stays on host (cheap, deterministic)
        # Adam moments, on the same backend as the weights
        mW = [xp.zeros_like(w, dtype=xp.float32) for w in self.W]
        vW = [xp.zeros_like(w, dtype=xp.float32) for w in self.W]
        mb = [xp.zeros_like(v, dtype=xp.float32) for v in self.b]
        vb = [xp.zeros_like(v, dtype=xp.float32) for v in self.b]
        t = 0
        n = int(X.shape[0])
        backend_note = "CUDA (GPU)" if xp is not np else "CPU"
        for ep in range(1, epochs + 1):
            order = xp.asarray(rng.permutation(n))
            loss_sum = 0.0
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                xb, yb = X[idx], y[idx]
                acts, p = self._forward(xb)
                eps = 1e-7
                loss_val = -xp.mean(yb * xp.log(p + eps) + (1 - yb) * xp.log(1 - p + eps))
                bs = int(idx.shape[0])
                loss_sum += self._to_float(loss_val) * bs
                d = (p - yb).astype(xp.float32) / bs
                t += 1
                for i in range(len(self.W) - 1, -1, -1):
                    a_prev = acts[i].astype(xp.float32)
                    gW = a_prev.T @ d
                    gb = d.sum(axis=0)
                    if i > 0:
                        d = (d @ self.W[i].astype(xp.float32).T) * (acts[i] > 0)
                    mW[i] = 0.9 * mW[i] + 0.1 * gW
                    vW[i] = 0.999 * vW[i] + 0.001 * (gW * gW)
                    mb[i] = 0.9 * mb[i] + 0.1 * gb
                    vb[i] = 0.999 * vb[i] + 0.001 * (gb * gb)
                    mhw = mW[i] / (1 - 0.9 ** t)
                    vhw = vW[i] / (1 - 0.999 ** t)
                    mhb = mb[i] / (1 - 0.9 ** t)
                    vhb = vb[i] / (1 - 0.999 ** t)
                    self.W[i] = (self.W[i].astype(xp.float32) -
                                 lr * mhw / (xp.sqrt(vhw) + 1e-8)).astype(dt)
                    self.b[i] = (self.b[i].astype(xp.float32) -
                                 lr * mhb / (xp.sqrt(vhb) + 1e-8)).astype(dt)
            if progress and (ep % max(1, epochs // 40) == 0 or ep == epochs):
                acc = self._to_float(xp.mean((self._predict_xp(X) > 0.5) == (y.ravel() > 0.5)))
                progress(f"Deep AI [{backend_note}]: epoch {ep}/{epochs} - "
                         f"loss {loss_sum / n:.4f} - accuracy {acc * 100:.1f}%")
            if should_stop and should_stop():
                break
        self.accuracy = self._to_float(
            xp.mean((self._predict_xp(X) > 0.5) == (y.ravel() > 0.5)))
        self.samples = int(n)
        self.trained = time.strftime("%Y-%m-%d %H:%M")
        self.enabled = True
        return self.accuracy

    # -- persistence -------------------------------------------------------
    def save(self):
        BQ_DIR.mkdir(parents=True, exist_ok=True)
        if self.W:
            # weights may live on the GPU; always persist plain host NumPy
            arrs = {f"W{i}": self._to_host(w) for i, w in enumerate(self.W)}
            arrs.update({f"b{i}": self._to_host(v) for i, v in enumerate(self.b)})
            np.savez_compressed(BANDNET_PATH, **arrs)
        BANDNET_META.write_text(json.dumps({
            "cfg": self.cfg, "enabled": self.enabled, "trained": self.trained,
            "samples": self.samples, "accuracy": self.accuracy,
            "params": self.params, "depth": len(self.W),
        }, indent=2))

    @classmethod
    def load(cls) -> "DeepBandNet":
        net = cls()
        try:
            meta = json.loads(BANDNET_META.read_text())
            net.cfg.update(meta.get("cfg", {}))
            net.enabled = bool(meta.get("enabled", False))
            net.trained = meta.get("trained", "")
            net.samples = int(meta.get("samples", 0))
            net.accuracy = float(meta.get("accuracy", 0.0))
            net.params = int(meta.get("params", 0))
            z = np.load(BANDNET_PATH)
            depth = int(meta.get("depth", 0))
            xp = net._array_module()
            # weights are stored on host; move them onto the active backend
            net.W = [xp.asarray(z[f"W{i}"]) for i in range(depth)]
            net.b = [xp.asarray(z[f"b{i}"]) for i in range(depth)]
        except Exception:
            net.W, net.b = [], []
            net.enabled = False
        return net

    def reset(self):
        self.__init__(self.cfg)
        for p in (BANDNET_PATH, BANDNET_META):
            try:
                p.unlink()
            except Exception:
                pass

    # -- use ---------------------------------------------------------------
    def ready(self) -> bool:
        return bool(self.W)

    def score_box(self, sig: np.ndarray, box) -> float:
        v = _patch_vector(sig, box, int(self.cfg["patch"]))
        if v is None or not self.W:
            return 1.0
        return float(self.predict(v[None, :])[0])

    def summary(self) -> str:
        gpu_note = ("CUDA/GPU" if GPU and _cp is not None else "CPU only (no CUDA GPU detected)")
        if not self.ready():
            return ("Deep AI: not trained yet - mark background, add training "
                    f"examples, then press 'Train deep AI'. ({gpu_note})")
        state = "ON" if self.enabled else "off"
        backend = "CUDA (GPU)" if self.gpu_active() else "CPU"
        return (f"Deep AI [{state}, {backend}]: {self.cfg['layers']} layers x {self.cfg['width']} "
                f"neurons, {self.params:,} weights @ {self.cfg['bits']}-bit, "
                f"{self.cfg['epochs']} epochs, {self.samples} patches, "
                f"training accuracy {self.accuracy * 100:.1f}% ({self.trained})")


DEEP_NET = DeepBandNet.load()


def _rect_overlap(a, b) -> float:
    """Fraction of box a that lies inside box b (both x0,x1,y0,y1)."""
    iw = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    area = max(1, (a[1] - a[0]) * (a[3] - a[2]))
    return (iw * ih) / area


def build_deep_dataset(patch: int, progress=None):
    """Positives = your ROIs. Negatives = painted background + random patches."""
    X, y = [], []
    for i, p in enumerate(list_training_samples(), 1):
        if progress:
            progress(f"Deep AI: reading example {i} ...")
        try:
            z = np.load(p)
            boxes = [tuple(int(v) for v in b) for b in z["boxes"]
                     if b[1] > b[0] + 1 and b[3] > b[2] + 1]
            bgs = ([tuple(int(v) for v in b) for b in z["bg"]]
                   if "bg" in z.files else [])
            sig = _prep_signal(z["gray"].astype(np.float32),
                               int(z["bg_radius"]), bool(int(z["dark"])))
        except Exception:
            continue
        h, w = sig.shape[:2]
        for box in boxes:
            v = _patch_vector(sig, box, patch)
            if v is not None:
                X.append(v)
                y.append(1.0)
        rng = np.random.default_rng(len(X) + 7)
        widths = [b[1] - b[0] for b in boxes] or [20]
        heights = [b[3] - b[2] for b in boxes] or [12]
        # 1) everything you painted as background, chopped into ROI-sized tiles
        for bx0, bx1, by0, by1 in bgs:
            bw = max(4, int(np.median(widths)))
            bh = max(4, int(np.median(heights)))
            for xx in range(bx0, max(bx0 + 1, bx1 - bw + 1), max(4, bw // 2)):
                for yy in range(by0, max(by0 + 1, by1 - bh + 1), max(3, bh // 2)):
                    v = _patch_vector(sig, (xx, xx + bw, yy, yy + bh), patch)
                    if v is not None:
                        X.append(v)
                        y.append(0.0)
        # 2) random patches that do not touch any accepted ROI
        target = max(len(boxes) * 3, 60)
        tries = 0
        made = 0
        while made < target and tries < target * 40:
            tries += 1
            bw = int(max(4, rng.choice(widths) * rng.uniform(0.7, 1.4)))
            bh = int(max(4, rng.choice(heights) * rng.uniform(0.7, 1.4)))
            if bw >= w - 2 or bh >= h - 2:
                continue
            xx = int(rng.integers(0, w - bw - 1))
            yy = int(rng.integers(0, h - bh - 1))
            cand = (xx, xx + bw, yy, yy + bh)
            if any(_iou(cand, b) > 0.05 for b in boxes):
                continue
            v = _patch_vector(sig, cand, patch)
            if v is not None:
                X.append(v)
                y.append(0.0)
                made += 1
    if not X:
        return None, None
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_deep_net(net: DeepBandNet, cfg: dict, progress=None) -> tuple[bool, str]:
    net.cfg.update({k: cfg[k] for k in DEEP_DEFAULTS if k in cfg})
    X, y = build_deep_dataset(int(net.cfg["patch"]), progress=progress)
    if X is None:
        return False, ("No training examples yet. Get the ROIs right on a blot, "
                       "paint the areas the AI should ignore, then press "
                       "'Add training example'.")
    if len(set(y.tolist())) < 2:
        return False, "Need both bands and background in the training data."
    backend_note = "CUDA (GPU)" if net.gpu_active() else "CPU"
    if net.cfg.get("use_gpu", True) and not (GPU and _cp is not None):
        backend_note = "CPU (no CUDA GPU detected - falling back)"
    if progress:
        progress(f"Deep AI [{backend_note}]: {len(X)} patches, building "
                 f"{net.cfg['layers']}x{net.cfg['width']} network ...")
    acc = net.fit(X, y, progress=progress)
    net.save()
    pos = int(y.sum())
    return True, (f"Deep AI trained on {backend_note}: {net.cfg['layers']} layers x "
                  f"{net.cfg['width']} neurons ({net.params:,} weights, "
                  f"{net.cfg['bits']}-bit), {net.cfg['epochs']} epochs on {len(X)} "
                  f"patches ({pos} bands / {len(X) - pos} background) - "
                  f"accuracy {acc * 100:.1f}%.")


def tighten_box(sig: np.ndarray, x0: int, x1: int, y0: int, y1: int,
                frac: float | None = None, coverage: float | None = None,
                pad: int | None = None, pad_frac: float | None = None,
                edge_frac: float | None = None,
                xlim: tuple[int, int] | None = None,
                ylim: tuple[int, int] | None = None):
    """Fit the ROI onto the band so it encapsulates ~all of the band signal.

    Alternates vertical/horizontal profile passes (each using the
    threshold + 99%-coverage criterion), then adds a pad that scales with the
    band size, and finally runs a containment guard that keeps growing the box
    while its border still carries band signal.  Erring outward is safe for
    quantification because the ramp background correction subtracts the extra
    background, while clipping a tail is not recoverable.
    """
    _p = ROI_MODEL.active()
    frac = _p["frac"] if frac is None else frac
    coverage = _p["coverage"] if coverage is None else coverage
    pad = _p["pad"] if pad is None else pad
    pad_frac = _p["pad_frac"] if pad_frac is None else pad_frac
    edge_frac = _p["edge_frac"] if edge_frac is None else edge_frac

    h, w = sig.shape[:2]
    xa, xb = (0, w) if xlim is None else (max(0, xlim[0]), min(w, xlim[1]))
    ya, yb = (0, h) if ylim is None else (max(0, ylim[0]), min(h, ylim[1]))
    x0, x1 = int(np.clip(x0, xa, xb - 1)), int(np.clip(x1, xa + 1, xb))
    y0, y1 = int(np.clip(y0, ya, yb - 1)), int(np.clip(y1, ya + 1, yb))

    # remember the seed so the fit can never run away and swallow a neighbour
    seed_cx, seed_cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    max_hw = max(0.5 * (x1 - x0) * 3.5, (x1 - x0) / 2.0 + 8.0)
    max_hh = max(0.5 * (y1 - y0) * 3.5, (y1 - y0) / 2.0 + 8.0)

    for _ in range(4):
        # --- vertical extent from the row profile inside the current x span
        col = gaussian_filter1d(sig[ya:yb, x0:x1].mean(axis=1), 1.0)
        c = int(np.clip((y0 + y1) / 2 - ya, 0, max(col.size - 1, 0)))
        c = int(np.clip(np.argmax(col[max(0, c - (y1 - y0)):c + (y1 - y0) + 1])
                        + max(0, c - (y1 - y0)), 0, max(col.size - 1, 0)))
        lo, hi = _extent_from_profile(col, c, frac, coverage)
        ny0, ny1 = ya + lo, ya + hi

        # --- horizontal extent from the column profile inside the new y span
        rowp = gaussian_filter1d(sig[ny0:ny1, xa:xb].mean(axis=0), 1.0)
        cxi = int(np.clip((x0 + x1) / 2 - xa, 0, max(rowp.size - 1, 0)))
        span = max(3, (x1 - x0) // 2)
        s = max(0, cxi - span)
        cxi = int(np.clip(int(np.argmax(rowp[s:cxi + span + 1])) + s,
                          0, max(rowp.size - 1, 0)))
        lo, hi = _extent_from_profile(rowp, cxi, frac, coverage)
        nx0, nx1 = xa + lo, xa + hi

        if (nx1 - nx0) < 3 or (ny1 - ny0) < 2:
            break
        if (nx0, nx1, ny0, ny1) == (x0, x1, y0, y1):
            break
        x0, x1, y0, y1 = nx0, nx1, ny0, ny1

    px = max(pad, int(round(pad_frac * (x1 - x0))))
    py = max(pad, int(round(pad_frac * (y1 - y0))))
    x0 = max(xa, x0 - px); x1 = min(xb, x1 + px)
    y0 = max(ya, y0 - py); y1 = min(yb, y1 + py)

    # --- containment guard: no band signal may remain on the ROI border ----
    box_peak = float(sig[y0:y1, x0:x1].max()) if (x1 > x0 and y1 > y0) else 0.0
    edge_thr = edge_frac * box_peak
    guard = 0
    while box_peak > 0 and guard < 12:
        grew = False
        if y0 > ya and float(sig[y0, x0:x1].mean()) > edge_thr:
            y0 -= 1; grew = True
        if y1 < yb and float(sig[y1 - 1, x0:x1].mean()) > edge_thr:
            y1 += 1; grew = True
        if x0 > xa and float(sig[y0:y1, x0].mean()) > edge_thr:
            x0 -= 1; grew = True
        if x1 < xb and float(sig[y0:y1, x1 - 1].mean()) > edge_thr:
            x1 += 1; grew = True
        if not grew:
            break
        guard += 1

    # --- runaway guard: stay within a sane multiple of the seed box --------
    x0 = int(max(x0, np.floor(seed_cx - max_hw)))
    x1 = int(min(x1, np.ceil(seed_cx + max_hw)))
    y0 = int(max(y0, np.floor(seed_cy - max_hh)))
    y1 = int(min(y1, np.ceil(seed_cy + max_hh)))
    x0, x1 = max(xa, x0), min(xb, max(x1, x0 + 1))
    y0, y1 = max(ya, y0), min(yb, max(y1, y0 + 1))

    return int(x0), int(x1), int(y0), int(y1)



# --------------------------------------------------------------------------
# 7. quantification
# --------------------------------------------------------------------------

def quantify(sig: np.ndarray, x0, x1, y0, y1):
    box = sig[y0:y1, x0:x1]
    if box.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    raw = float(box.sum())
    pad = max(3, (y1 - y0))
    top = sig[max(0, y0 - pad):y0, x0:x1]
    bot = sig[y1:min(sig.shape[0], y1 + pad), x0:x1]
    if top.size and bot.size:
        t = float(np.median(top))
        bo = float(np.median(bot))
        rows = np.linspace(t, bo, y1 - y0)[:, None]
        bg_img = np.repeat(rows, x1 - x0, axis=1)
    else:
        ref = top if top.size else bot
        bg_img = np.full_like(box, float(np.median(ref)) if ref.size else 0.0)
    bg = float(bg_img.sum())
    return raw, bg, max(raw - bg, 0.0), float(box.max())


def lane_profile_auc(sig: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float:
    """Area under the 1-D lane intensity 'hill' for this band - i.e. the exact
    same quantity the intensity-profile popup shades in orange. This is the
    mean-across-lane-width profile, integrated (trapezoidal, unit spacing)
    over the band's y-extent above the local (flanking) baseline.
    """
    h, w = sig.shape[:2]
    x0c, x1c = max(0, int(x0)), min(w, int(x1))
    y0i, y1i = int(y0), int(y1)
    if x1c <= x0c or y1i <= y0i:
        return 0.0
    pad = max(20, int((y1i - y0i) * 3))
    ya, yb = max(0, y0i - pad), min(h, y1i + pad)
    if yb <= ya:
        return 0.0
    strip = sig[ya:yb, x0c:x1c]
    if strip.size == 0:
        return 0.0
    prof = strip.mean(axis=1)
    prof = gaussian_filter1d(prof.astype(np.float64), 1.0)
    ys = np.arange(ya, yb)
    inside = (ys >= y0i) & (ys < y1i)
    outside_vals = prof[~inside]
    baseline = float(np.median(outside_vals)) if outside_vals.size else float(np.min(prof))
    band_vals = prof[inside]
    if band_vals.size == 0:
        return 0.0
    return float(np.clip(band_vals - baseline, 0, None).sum())


def measure_roi(sig: np.ndarray, x0: int, y0: int, x1: int, y1: int, min_snr: float = 2.5):
    """Full measurement of an arbitrary rectangle: volumes, peak, centre, sigma, SNR."""
    h, w = sig.shape[:2]
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
    raw, bg, net, pk = quantify(sig, x0, x1, y0, y1)
    auc = lane_profile_auc(sig, x0, x1, y0, y1)

    prof = sig[y0:y1, x0:x1].mean(axis=1) if y1 > y0 else np.array([0.0])
    ys = np.arange(y0, y1, dtype=float)
    base = float(prof.min()) if prof.size else 0.0
    wts = np.clip(prof - base, 0, None)
    if wts.sum() > 0:
        centre = float((ys * wts).sum() / wts.sum())
        sigma = float(np.sqrt(max(((ys - centre) ** 2 * wts).sum() / wts.sum(), 1e-6)))
    else:
        centre, sigma = float((y0 + y1) / 2.0), max((y1 - y0) / 4.0, 1.0)

    pad = max(4, (y1 - y0))
    ring = np.concatenate([
        sig[max(0, y0 - pad):y0, x0:x1].ravel(),
        sig[y1:min(h, y1 + pad), x0:x1].ravel(),
    ]) if x1 > x0 else np.array([0.0])
    if ring.size:
        noise = 1.4826 * float(np.median(np.abs(ring - np.median(ring)))) + 1e-6
        snr = float((pk - float(np.median(ring))) / noise)
    else:
        snr = 0.0
    return dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, raw_volume=raw, bg_volume=bg,
                net_volume=net, peak=pk, centre_y=centre, sigma=sigma,
                snr=snr, empty=snr < min_snr, auc=auc)


# --------------------------------------------------------------------------
# 7b. ROI de-overlap  (new)
# --------------------------------------------------------------------------
def _deoverlap_row(row_bands: list[Band], sig: np.ndarray) -> None:
    """Trim adjacent-lane ROI boxes so they never overlap each other.

    Bands that belong to the SAME lane (stacked vertically) are left alone -
    it's normal for them to share the lane's full width. Bands in DIFFERENT
    lanes that still intersect after tighten_box (e.g. because both grew into
    the padding on either side of the lane gap) have their shared region
    split down the middle, one half going to each box. Anything that actually
    changed size gets re-measured (raw/bg/net volume, peak, AUC) so the
    numbers stay consistent with the trimmed footprint.
    """
    if len(row_bands) < 2:
        return
    changed: set[int] = set()
    for _ in range(3):  # a couple of passes lets a chain of 3+ overlaps settle
        ordered = sorted(row_bands, key=lambda b: b.x + b.w / 2.0)
        touched = False
        for a, b in zip(ordered, ordered[1:]):
            if a.lane == b.lane:
                continue
            ax0, ax1 = a.x, a.x + a.w
            bx0, bx1 = b.x, b.x + b.w
            ay0, ay1 = a.y, a.y + a.h
            by0, by1 = b.y, b.y + b.h
            ov_x = min(ax1, bx1) - max(ax0, bx0)
            ov_y = min(ay1, by1) - max(ay0, by0)
            if ov_x <= 0 or ov_y <= 0:
                continue  # boxes don't actually intersect - leave them be
            mid = int(round((ax1 + bx0) / 2.0))
            mid = max(ax0 + 1, min(mid, bx1 - 1))
            if mid != ax1:
                a.w = max(1, mid - a.x)
                changed.add(id(a))
            if mid != bx0:
                b.w = max(1, bx1 - mid)
                b.x = mid
                changed.add(id(b))
            touched = True
        if not touched:
            break
    for b in row_bands:
        if id(b) not in changed:
            continue
        raw, bg, net, pk = quantify(sig, b.x, b.x + b.w, b.y, b.y + b.h)
        b.raw_volume, b.bg_volume, b.net_volume, b.peak = raw, bg, net, pk
        b.auc = lane_profile_auc(sig, b.x, b.x + b.w, b.y, b.y + b.h)


# --------------------------------------------------------------------------
# 8. pipeline
# --------------------------------------------------------------------------
def detect_bands(gray: np.ndarray, sensitivity: float = 1.0, bg_radius: int = 30,
                 dark_bands: bool = True, deskew: bool = True,
                 fill_grid: bool = True, min_snr: float = 2.5) -> Result:
    sig0 = to_density(gray, dark_bands)
    angle = estimate_skew(rolling_ball(gblur(sig0, 1.0), bg_radius)) if deskew else 0.0
    g = rotate(gray, angle) if angle else gray
    sig = to_density(g, dark_bands)
    sig = gblur(sig, 0.8)
    sig = rolling_ball(sig, bg_radius)

    bands: list[Band] = []
    rows = find_rows(sig)

    per_row = []
    for row in rows:
        centres, pitch, prof = find_lanes(sig, row, sensitivity, fill_grid=fill_grid)
        per_row.append((row, centres, pitch, prof))
    pitches = [p for _, _, p, _ in per_row if p > 0]
    pitch = float(np.median(pitches)) if pitches else 0.0
    shared = (consensus_centres([c for _, c, _, _ in per_row], pitch)
              if fill_grid and len(per_row) > 1 else None)

    for ri, (row, centres, _p, prof) in enumerate(per_row, start=1):
        use = shared if shared is not None and len(shared) else centres
        if use is None or len(use) == 0:
            continue
        lanes = lane_boxes(use, prof, pitch, sig.shape[1])
        row_bands: list[Band] = []
        rpad = max(8, row[1] - row[0])
        ylim = (max(0, row[0] - rpad), min(sig.shape[0], row[1] + rpad))
        for li, (x0, x1) in enumerate(lanes, start=1):
            segs = find_bands_in_lane(sig, x0, x1, row[0], row[1], sensitivity)
            if not segs:
                continue
            # Give the ROI fitter room beyond the lane's own boundary box: the
            # lane box's right/left edges come from the mid-point valley to the
            # NEXT lane (and are further capped to +-0.5*pitch), which is often
            # short of where the band signal actually decays.  Using the lane
            # box itself as tighten_box's xlim hard-clamps the fit there, so
            # the right side of asymmetric/smeared bands was silently cut off.
            # The runaway guard inside tighten_box (based on the seed box
            # size) still keeps the fit from spilling into a neighbouring
            # lane, so this padding is safe. Any residual overlap between
            # neighbouring lanes' boxes that slips through is cleaned up by
            # _deoverlap_row() below, once all of this row's bands exist.
            lane_w = x1 - x0
            xpad = max(4, int(round(lane_w * 0.4)))
            xlim = (max(0, x0 - xpad), min(sig.shape[1], x1 + xpad))
            for bi, (y0, y1, cy, sd, snr) in enumerate(segs, start=1):
                # shrink-wrap the lane-wide box onto the actual band footprint
                bx0, bx1, by0, by1 = tighten_box(sig, x0, x1, y0, y1,
                                                 xlim=xlim, ylim=ylim)
                raw, bg, net, pk = quantify(sig, bx0, bx1, by0, by1)
                if snr < min_snr and net <= 0:
                    continue
                auc_val = lane_profile_auc(sig, bx0, bx1, by0, by1)
                nb = Band(ri, li, bi, bx0, by0, bx1 - bx0, by1 - by0,
                         raw, bg, net, pk, cy, sd, snr,
                         empty=snr < min_snr)
                nb.auc = auc_val
                row_bands.append(nb)
        # ROIs from different lanes must never overlap each other; trim any
        # that still do (see _deoverlap_row for why this can happen).
        _deoverlap_row(row_bands, sig)
        # continuous lane numbering per row (L1..Ln left-to-right, gaps ignored)
        order = sorted({b.lane for b in row_bands})
        remap = {old: new for new, old in enumerate(order, start=1)}
        for b in row_bands:
            b.lane = remap[b.lane]
        bands.extend(row_bands)
    return Result(bands=bands, density=sig, gray=g, angle=angle)



# ==========================================================================
# NORMALIZATION ENGINE
# ==========================================================================
MODES = [
    "AUC / integrated density (raw)",
    "% of reference band",
    "% of reference lane (same band #)",
    "Loading control ratio (target / control row)",
    "Fold change vs control lane",
    "% of lane total",
    "% of row total",
    "Relative to row mean",
    "Total-protein normalized",
    "Same lane across rows (row / reference row)",
]


@dataclass
class NormConfig:
    mode: str = "AUC / integrated density (raw)"
    ref_row: int = 1          # reference band coordinates
    ref_lane: int = 1
    ref_band: int = 1
    ctrl_row: int = 2         # loading control / total protein row
    ctrl_band: int = 1        # which band inside the control row
    ctrl_lane: int = 1        # the lane that becomes 1.00 in fold-change modes
    exclude_empty: bool = True


class Normalizer:
    """Turns a list of Band into normalized values + traceable denominators."""

    def __init__(self, bands: list[Band], cfg: NormConfig):
        self.bands = bands
        self.cfg = cfg
        self.by_key = {b.key: b for b in bands}
        self.rows = sorted({b.row for b in bands})
        self.lanes = sorted({b.lane for b in bands})
        self.indices = sorted({b.index for b in bands})

    # ---- helpers ---------------------------------------------------------
    def _usable(self, b: Band) -> bool:
        return not (self.cfg.exclude_empty and b.empty)

    def value(self, row: int, lane: int, index: int) -> float:
        b = self.by_key.get((row, lane, index))
        return float(b.auc) if b else 0.0

    def lane_total(self, row: int, lane: int) -> float:
        return float(sum(b.auc for b in self.bands
                         if b.row == row and b.lane == lane and self._usable(b)))

    def row_total(self, row: int) -> float:
        return float(sum(b.auc for b in self.bands
                         if b.row == row and self._usable(b)))

    def row_mean(self, row: int) -> float:
        vals = [b.auc for b in self.bands if b.row == row and self._usable(b)]
        return float(np.mean(vals)) if vals else 0.0

    def _loading_ratio(self, lane: int, row: int, index: int) -> tuple[float, float]:
        den = self.value(self.cfg.ctrl_row, lane, self.cfg.ctrl_band)
        num = self.value(row, lane, index)
        return (num / den if den > 0 else 0.0), den

    def _total_protein_ratio(self, lane: int, row: int, index: int) -> tuple[float, float]:
        den = self.lane_total(self.cfg.ctrl_row, lane)
        num = self.value(row, lane, index)
        return (num / den if den > 0 else 0.0), den

    # ---- main ------------------------------------------------------------
    def compute(self) -> dict[tuple[int, int, int], tuple[float, float, str]]:
        c = self.cfg
        m = c.mode
        out: dict[tuple[int, int, int], tuple[float, float, str]] = {}

        if m == "% of reference band":
            den = self.value(c.ref_row, c.ref_lane, c.ref_band)
            desc = f"R{c.ref_row} L{c.ref_lane} B{c.ref_band}"
            for b in self.bands:
                out[b.key] = (100.0 * b.auc / den if den > 0 else 0.0, den, desc)

        elif m == "% of reference lane (same band #)":
            for b in self.bands:
                den = self.value(b.row, c.ref_lane, b.index)
                if den <= 0:
                    den = self.lane_total(b.row, c.ref_lane)
                out[b.key] = (100.0 * b.auc / den if den > 0 else 0.0, den,
                              f"R{b.row} L{c.ref_lane} B{b.index}")

        elif m == "Loading control ratio (target / control row)":
            for b in self.bands:
                r, den = self._loading_ratio(b.lane, b.row, b.index)
                out[b.key] = (r, den, f"R{c.ctrl_row} L{b.lane} B{c.ctrl_band}")

        elif m == "Fold change vs control lane":
            base, _ = self._loading_ratio(c.ctrl_lane, c.ref_row, c.ref_band)
            for b in self.bands:
                r, den = self._loading_ratio(b.lane, b.row, b.index)
                ref_r, _ = self._loading_ratio(c.ctrl_lane, b.row, b.index)
                scale = ref_r if ref_r > 0 else base
                out[b.key] = (r / scale if scale > 0 else 0.0, den,
                              f"(R{b.row}/R{c.ctrl_row}) / lane {c.ctrl_lane}")

        elif m == "% of lane total":
            for b in self.bands:
                den = self.lane_total(b.row, b.lane)
                out[b.key] = (100.0 * b.auc / den if den > 0 else 0.0, den,
                              f"lane total R{b.row} L{b.lane}")

        elif m == "% of row total":
            for b in self.bands:
                den = self.row_total(b.row)
                out[b.key] = (100.0 * b.auc / den if den > 0 else 0.0, den,
                              f"row total R{b.row}")

        elif m == "Relative to row mean":
            for b in self.bands:
                den = self.row_mean(b.row)
                out[b.key] = (b.auc / den if den > 0 else 0.0, den,
                              f"row mean R{b.row}")

        elif m == "Same lane across rows (row / reference row)":
            # Compares the SAME lane (and same band number) between blot rows:
            # e.g. row 2 / lane 2 divided by row 1 / lane 2.
            for b in self.bands:
                den = self.value(c.ref_row, b.lane, b.index)
                if den <= 0:
                    den = self.lane_total(c.ref_row, b.lane)
                out[b.key] = (b.auc / den if den > 0 else 0.0, den,
                              f"R{c.ref_row} L{b.lane} B{b.index}")

        elif m == "Total-protein normalized":
            ref_lane_vals = {}
            for b in self.bands:
                r, den = self._total_protein_ratio(b.lane, b.row, b.index)
                key = (b.row, b.index)
                if key not in ref_lane_vals:
                    ref_lane_vals[key] = self._total_protein_ratio(
                        c.ctrl_lane, b.row, b.index)[0]
                scale = ref_lane_vals[key]
                out[b.key] = (r / scale if scale > 0 else r, den,
                              f"total protein R{c.ctrl_row} L{b.lane}"
                              + (f" / lane {c.ctrl_lane}" if scale > 0 else ""))

        else:  # "AUC / integrated density (raw)" - the un-normalized measure
            for b in self.bands:
                out[b.key] = (b.auc, 1.0, "none")

        return out

    def compute_all(self) -> dict[tuple[int, int, int], dict[str, float]]:
        """Run every normalization mode in MODES against this same bands/cfg
        (reference/control selections held fixed) and return, per band, a
        dict of {mode_name: value}. This is the "full list of result types"
        used for statistics / export - it does not change self.cfg.mode
        permanently."""
        orig_mode = self.cfg.mode
        per_mode: dict[str, dict[tuple[int, int, int], tuple[float, float, str]]] = {}
        try:
            for m in MODES:
                self.cfg.mode = m
                per_mode[m] = self.compute()
        finally:
            self.cfg.mode = orig_mode
        out: dict[tuple[int, int, int], dict[str, float]] = {}
        for b in self.bands:
            out[b.key] = {m: per_mode[m].get(b.key, (0.0, 0.0, ""))[0] for m in MODES}
        return out

    def unit(self) -> str:
        m = self.cfg.mode
        if m.startswith("%"):
            return "%"
        if m == "AUC / integrated density (raw)":
            return "a.u."
        return "x"


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class WBDetectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WB Detect v3.8" + ("  [GPU]" if GPU else ""))
        self._size_window()
        self._init_theme()
        self.image_path = None
        self.original_img = None
        self.result = Result()
        self.norm: dict = {}
        self._detected_backup: list[Band] = []

        # ROI editor state
        self.scale = 1.0        # effective image -> canvas scale
        self.fit_scale = 1.0    # scale that fits the whole image in the viewport
        self.zoom_mode = "fit"  # "fit" or a numeric zoom factor
        self._disp_gray = None
        self.selected: Band | None = None
        self._drag = None          # dict describing the current drag
        self.tool = tk.StringVar(value="select")
        # areas you painted as background - the AI ignores anything inside them
        self.bg_regions: list[tuple[int, int, int, int]] = []
        self._bg_photo = None
        self.ox = 0.0          # canvas offset that centers the image horizontally
        self.oy = 0.0          # canvas offset that centers the image vertically
        self._ew = 0           # scroll-region (world) width
        self._eh = 0           # scroll-region (world) height

        # multi-select state: ids of Band objects currently marquee-selected
        self.multi_sel: set[int] = set()

        # intensity-profile popup ("hill" plot, like Fiji's Plot Lanes)
        self.profile_win: tk.Toplevel | None = None
        self.profile_canvas: tk.Canvas | None = None
        self._profile_data = None

        # background training state
        self._train_queue: "queue.Queue" = queue.Queue()
        self._train_thread: threading.Thread | None = None

        # ---- detection toolbar -------------------------------------------
        top = ttk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        ttk.Button(top, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Run Detection", command=self.run_detection).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Export Excel", command=self.export_excel).pack(side=tk.LEFT, padx=4)

        # Detection defaults (advanced sliders removed from the main toolbar).
        self.sens = tk.DoubleVar(value=1.0)
        self.bgr = tk.IntVar(value=30)

        self.dark = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Dark bands", variable=self.dark,
                        command=self.run_detection).pack(side=tk.LEFT, padx=8)
        self.deskew = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Auto-deskew", variable=self.deskew,
                        command=self.run_detection).pack(side=tk.LEFT, padx=8)
        self.grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Lane grid", variable=self.grid,
                        command=self.run_detection).pack(side=tk.LEFT, padx=8)
        gpu_txt = "GPU: CUDA detected" if GPU else "GPU: no CUDA device found (CPU only)"
        ttk.Label(top, text=gpu_txt,
                  foreground="#2e7d32" if GPU else "#b71c1c").pack(side=tk.LEFT, padx=10)

        # ---- compact upper controls: editing/AI left, normalization right -
        upper = ttk.Frame(root)
        upper.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 4))
        upper.columnconfigure(0, weight=44)
        upper.columnconfigure(1, weight=56)

        upper_left = ttk.Frame(upper)
        upper_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # ---- ROI editor hint -----------------------------------------------
        # The tool switcher (Select/Draw/Multi-select) now lives right above
        # the canvas so it's quick to reach; background marking moved into
        # the AI panel below, since it's AI training data.
        rb = ttk.LabelFrame(upper_left, text="ROI editor")
        rb.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        # Selected ROI coordinates remain synchronized internally.
        self.sel_row = tk.IntVar(value=1)
        self.sel_lane = tk.IntVar(value=1)
        self.sel_band = tk.IntVar(value=1)

        self.roi_hint = ttk.Label(
            rb,
            text="Click/drag to edit · Shift+drag to create · Del removes · "
                 "Wheel zooms · middle-drag pans · F fits · "
                 "tool buttons are above the canvas · "
                 "'Multi-select' + drag a box (Shift adds) picks several ROIs, "
                 "then Del removes them all · "
                 "background marking (for AI training) is in the AI panel below · "
                 "click the 📈 icon on a results row to open its intensity plot",
            foreground="#555", wraplength=560)
        self.roi_hint.pack(side=tk.TOP, anchor=tk.W, padx=8, pady=(3, 3))

        # ---- AI-assisted detection (one unified model, foldable) ----------
        # Two engines work together behind a single switch: a small model
        # that learns how to shrink-wrap ROIs onto your corrected boxes
        # (geometry), and a deep classifier that learns to recognise real
        # bands vs. background (existence). They share the same training
        # examples, the same on/off switch, and train together in one pass.
        # The deep classifier's math (see DeepBandNet) runs on CUDA via CuPy
        # whenever "Use GPU (CUDA)" below is checked and a GPU is present.
        # Background marking lives here too, since it's AI training data.
        ai = ttk.LabelFrame(upper_left, text="")
        ai.pack(side=tk.TOP, fill=tk.X)
        ai_header = ttk.Frame(ai)
        ai_header.pack(side=tk.TOP, fill=tk.X)
        self.ai_toggle_btn = ttk.Button(ai_header, text="\u25b6 AI-assisted detection",
                                        command=self.toggle_ai_panel)
        self.ai_toggle_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.ai_body = ttk.Frame(ai)  # shown/hidden by toggle_ai_panel

        a0 = ttk.Frame(self.ai_body)
        a0.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(3, 2))

        self.use_ai = tk.BooleanVar(value=ROI_MODEL.enabled or DEEP_NET.enabled)
        ttk.Checkbutton(a0, text="Use AI", variable=self.use_ai,
                        command=self.toggle_use_ai).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(a0, text="Add training example",
                   command=self.add_training_example).pack(side=tk.LEFT, padx=2)
        self.train_btn = ttk.Button(a0, text="Train AI", command=self.train_ai)
        self.train_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(a0, text="Auto-fit selected",
                   command=self.autofit_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(a0, text="Auto-fit all",
                   command=self.autofit_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(a0, text="Clean up ROIs now",
                   command=self.deep_cleanup).pack(side=tk.LEFT, padx=2)
        ttk.Button(a0, text="Reset AI",
                   command=self.reset_ai).pack(side=tk.LEFT, padx=2)

        # network shape for the classifier half - tucked into its own row so
        # the main controls above stay simple
        self.deep_layers = tk.IntVar(value=int(DEEP_NET.cfg["layers"]))
        self.deep_width = tk.IntVar(value=int(DEEP_NET.cfg["width"]))
        self.deep_epochs = tk.IntVar(value=int(DEEP_NET.cfg["epochs"]))
        self.deep_bits = tk.StringVar(value=str(DEEP_NET.cfg["bits"]))
        self.deep_lr = tk.DoubleVar(value=float(DEEP_NET.cfg["lr"]))
        self.deep_thr = tk.DoubleVar(value=float(DEEP_NET.cfg["threshold"]))
        # GPU checkbox: only meaningful (and only enabled) when a CUDA device
        # was actually detected at startup (module-level GPU flag / CuPy).
        self.deep_gpu = tk.BooleanVar(
            value=bool(DEEP_NET.cfg.get("use_gpu", True)) and GPU)

        d0 = ttk.Frame(self.ai_body)
        d0.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 2))
        ttk.Label(d0, text="Network:").pack(side=tk.LEFT, padx=(0, 4))
        for text, var, lo, hi, step, wdt in (
                ("Layers", self.deep_layers, 1, 64, 1, 5),
                ("Neurons/layer", self.deep_width, 8, 4096, 8, 6),
                ("Epochs", self.deep_epochs, 1, 20000, 25, 7),
                ("Learn rate", self.deep_lr, 0.0001, 0.05, 0.0005, 7),
                ("Ignore below", self.deep_thr, 0.05, 0.95, 0.05, 5)):
            ttk.Label(d0, text=text).pack(side=tk.LEFT, padx=(6, 2))
            ttk.Spinbox(d0, from_=lo, to=hi, increment=step, textvariable=var,
                        width=wdt).pack(side=tk.LEFT)
        ttk.Label(d0, text="Bits").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Combobox(d0, textvariable=self.deep_bits, values=("16", "32", "64"),
                     state="readonly", width=4).pack(side=tk.LEFT)
        gpu_chk_state = "normal" if (GPU and _cp is not None) else "disabled"
        gpu_chk_text = "Use GPU (CUDA)" if (GPU and _cp is not None) \
            else "Use GPU (CUDA) - not detected"
        ttk.Checkbutton(d0, text=gpu_chk_text, variable=self.deep_gpu,
                        state=gpu_chk_state).pack(side=tk.LEFT, padx=(10, 2))

        # Background marking (AI training data) sits at the bottom of the AI
        # panel, below the network settings, since it's the last thing you
        # touch before training rather than the first.
        bgrow = ttk.Frame(self.ai_body)
        bgrow.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 2))
        ttk.Radiobutton(bgrow, text="Mark background (ignore)", value="bg",
                        variable=self.tool, command=self.redraw).pack(side=tk.LEFT)
        ttk.Button(bgrow, text="Clear background marks",
                   command=self.clear_bg_regions).pack(side=tk.LEFT, padx=6)

        self.ai_lbl = ttk.Label(self.ai_body, text=self._combined_ai_summary(),
                                foreground="#555", wraplength=560,
                                justify=tk.LEFT)
        self.ai_lbl.pack(side=tk.TOP, anchor=tk.W, padx=8, pady=(0, 3))

        # ---- normalization, aligned at the very top right ----------------
        nb = ttk.LabelFrame(upper, text="Normalization")
        nb.grid(row=0, column=1, sticky="nsew")

        self.mode = tk.StringVar(value=MODES[0])
        self.excl = tk.BooleanVar(value=True)
        self.v_ref_row = tk.IntVar(value=1)
        self.v_ref_lane = tk.IntVar(value=1)
        self.v_ref_band = tk.IntVar(value=1)
        self.v_ctrl_row = tk.IntVar(value=2)
        self.v_ctrl_band = tk.IntVar(value=1)
        self.v_ctrl_lane = tk.IntVar(value=1)

        mode_row = ttk.Frame(nb)
        mode_row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 2))
        ttk.Label(mode_row, text="Mode").pack(side=tk.LEFT)
        self.mode_cb = ttk.Combobox(mode_row, textvariable=self.mode, values=MODES,
                                    state="readonly", width=34)
        self.mode_cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.mode_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_norm())

        ttk.Checkbutton(nb, text="Ignore below-SNR bands in totals",
                        variable=self.excl, command=self.apply_norm).pack(
                            side=tk.TOP, anchor=tk.W, padx=6, pady=2)

        pick_row = ttk.Frame(nb)
        pick_row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 4))
        ttk.Button(pick_row, text="Reference = selected ROI  (R)",
                   command=self.ref_from_selected).pack(side=tk.LEFT, fill=tk.X,
                                                        expand=True, padx=(0, 3))
        ttk.Button(pick_row, text="Control = selected ROI  (C)",
                   command=self.ctrl_from_selected).pack(side=tk.LEFT, fill=tk.X,
                                                         expand=True, padx=(3, 0))

        self.norm_hint = ttk.Label(nb, text="", foreground="#555", wraplength=700)
        self.norm_hint.pack(side=tk.TOP, anchor=tk.W, padx=8, pady=(0, 4))

        # ---- main area: canvas on left, results table on right ------------
        main = ttk.Frame(root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=6)

        # ---- canvas preview (left pane, fills remaining space) ------------
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew")

        # Tool switcher lives right above the canvas so it's fast to reach.
        tool_bar = ttk.Frame(left)
        tool_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Radiobutton(tool_bar, text="Select / edit", value="select",
                        variable=self.tool, command=self.redraw).pack(
                            side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(tool_bar, text="Draw new ROI", value="draw",
                        variable=self.tool, command=self.redraw).pack(
                            side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(tool_bar, text="Multi-select", value="multi",
                        variable=self.tool, command=self.redraw).pack(side=tk.LEFT)
        # Zoom remains available with the mouse wheel and keyboard shortcuts.
        self.zoom_lbl = ttk.Label(tool_bar, text="")
        self.zoom_lbl.pack(side=tk.RIGHT, padx=6)

        # ---- results table (right pane, same height as canvas) ------------
        right = ttk.LabelFrame(main, text="Results")
        right.grid(row=0, column=1, sticky="ns", padx=(6, 0))

        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        wrap = ttk.Frame(left)
        wrap.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(wrap, width=980, height=560,
                                background="#222", highlightthickness=0,
                                cursor="crosshair")
        vsb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.canvas.yview)
        hsb = ttk.Scrollbar(wrap, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        # scroll + zoom with the wheel (Ctrl/Cmd + wheel zooms)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_wheel)
        self.canvas.bind("<Control-MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        for k in ("<Delete>", "<BackSpace>"):
            self.canvas.bind(k, lambda e: self.delete_any_selected())
        for k in ("<r>", "<R>"):
            self.canvas.bind(k, lambda e: self.ref_from_selected())
        for k in ("<c>", "<C>"):
            self.canvas.bind(k, lambda e: self.ctrl_from_selected())
        for k, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            self.canvas.bind(f"<{k}>", lambda e, a=dx, b=dy: self.nudge(a, b))
            self.canvas.bind(f"<Shift-{k}>", lambda e, a=dx * 5, b=dy * 5: self.nudge(a, b))
        # middle-button drag pans the canvas
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, 1))
        # quick keys: F = fit, 0 = 100%
        self.canvas.bind("<Key-f>", lambda e: self.zoom_fit())
        self.canvas.bind("<Key-F>", lambda e: self.zoom_fit())
        self.canvas.bind("<Key-0>", lambda e: self.set_zoom(1.0))

        # ---- bottom status dock ------------------------------------------
        bottom = ttk.Frame(root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # ---- results table -----------------------------------------------
        # Tk's Treeview has one native heading row, so a precisely sized strip
        # above it supplies the higher-level scientific column groups.
        # "plot" is a dedicated icon column: clicking it (and ONLY it) opens
        # the intensity-profile popup for that row - selecting a row or an
        # ROI no longer opens it automatically.
        cols = ("row", "lane", "net", "auc", "peak", "snr", "norm", "basis", "plot")
        heads = ("Row", "Lane", "Net volume", "Hill AUC", "Peak", "SNR",
                 "Normalized", "Relative to", "📈")
        widths = (44, 44, 100, 100, 64, 58, 110, 160, 34)
        self._heads = dict(zip(cols, heads))
        self._plot_col_id = f"#{cols.index('plot') + 1}"

        style = ttk.Style()
        style.configure("Bench.Treeview", background="#FAF9F6", fieldbackground="#FAF9F6",
                        foreground="#242824", rowheight=25, borderwidth=0,
                        font=("Segoe UI", 9))
        style.configure("Bench.Treeview.Heading", background="#EAF0F2",
                        foreground="#242824", relief="flat", padding=(5, 6),
                        font=("Segoe UI", 9, "bold"))
        style.map("Bench.Treeview", background=[("selected", "#39734D")],
                  foreground=[("selected", "#FAF9F6")])
        style.map("Bench.Treeview.Heading", background=[("active", "#DCE4E0")])

        tf = ttk.Frame(right)
        tf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 8))

        table_bar = tk.Frame(tf, background="#FAF9F6", highlightbackground="#E4E0D8",
                             highlightthickness=1)
        table_bar.pack(side=tk.TOP, fill=tk.X)
        self.table_summary = tk.Label(table_bar, text="RESULTS", background="#FAF9F6",
                                      foreground="#39734D", font=("Segoe UI", 9, "bold"),
                                      padx=10, pady=5)
        self.table_summary.pack(side=tk.LEFT)
        tk.Label(table_bar, text="Click any column heading to sort · click 📈 for the profile",
                 background="#FAF9F6", foreground="#5E665F",
                 font=("Segoe UI", 8), padx=10).pack(side=tk.RIGHT)

        group_strip = tk.Frame(tf, background="#E0E9E7")
        group_strip.pack(side=tk.TOP, fill=tk.X)
        groups = (("BAND IDENTITY", sum(widths[0:2]), False),
                  ("RAW SIGNAL", sum(widths[2:6]), False),
                  ("NORMALIZED RESULT", sum(widths[6:8]), True),
                  ("", widths[8], False))
        for title, group_width, emphasized in groups:
            tk.Label(group_strip, text=title, width=1,
                     background="#39734D" if emphasized else "#E0E9E7",
                     foreground="#FAF9F6" if emphasized else "#526057",
                     font=("Segoe UI", 8, "bold"), anchor=tk.W, padx=8, pady=4,
                     highlightbackground="#E4E0D8", highlightthickness=1
                     ).pack(side=tk.LEFT, fill=tk.Y, ipadx=max(0, (group_width - 70) // 2))

        tree_wrap = ttk.Frame(tf)
        tree_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=20,
                                 style="Bench.Treeview")
        numeric = {"row", "lane", "net", "auc", "peak", "snr", "norm"}
        for c, t, width in zip(cols, heads, widths):
            self.tree.heading(c, text=t, command=lambda cc=c: self.sort_by(cc))
            self.tree.column(c, width=width, minwidth=28, stretch=(c != "plot"),
                             anchor=tk.CENTER if c == "plot"
                             else (tk.E if c in numeric else tk.W))
        self.tree.tag_configure("even", background="#FAF9F6")
        self.tree.tag_configure("odd", background="#EEF1EC")
        # one colour motif per blot row (green / blue / red / ...)
        for _i in range(len(ROW_COLORS)):
            self.tree.tag_configure(f"blotrow{_i + 1}", background=row_tint(_i + 1),
                                    foreground=row_color(_i + 1))
        self.tree.tag_configure("manual", foreground="#315F41")
        self.tree.tag_configure("refband", background="#FFF3C4", foreground="#5A4300")
        self.tree.tag_configure("ctrlband", background="#EDE7F6", foreground="#3F2D63")

        xsb = ttk.Scrollbar(tf, orient=tk.HORIZONTAL, command=self.tree.xview)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        tsb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set, xscrollcommand=xsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tsb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_table_select)
        self.tree.bind("<Button-1>", self.on_table_click, add="+")
        for _k in ("<r>", "<R>"):
            self.tree.bind(_k, lambda e: self.ref_from_selected())
        for _k in ("<c>", "<C>"):
            self.tree.bind(_k, lambda e: self.ctrl_from_selected())
        self._sort_desc = False

        self.status = ttk.Label(bottom, text="Load a blot image to begin.")
        self.status.pack(side=tk.BOTTOM, anchor=tk.W, padx=10, pady=4)
        self._job = None

    # ---- theme -------------------------------------------------------------
    def _init_theme(self):
        """A very subtle green tint across the window - sage/mint neutrals,
        not a loud green skin. Keeps the canvas dark (for image contrast) and
        the results table's existing forest-green accents untouched."""
        BG = "#F3F7F4"        # near-white with a whisper of green
        BG_ALT = "#EAF1EC"    # slightly deeper tint, for panels/labelframes
        ACCENT = "#39734D"    # existing forest green used by the results table
        BORDER = "#D8E6DC"

        self.root.configure(background=BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG)
        style.configure("TCheckbutton", background=BG)
        style.configure("TRadiobutton", background=BG)
        style.configure("TLabelframe", background=BG_ALT, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=BG_ALT, foreground=ACCENT,
                        font=("Segoe UI", 9, "bold"))
        style.configure("TButton", background=BG_ALT, bordercolor=BORDER)
        style.map("TButton", background=[("active", "#DCEBE1")])
        style.configure("TCombobox", fieldbackground="#FFFFFF")
        style.configure("TNotebook", background=BG)

    # ---- detection --------------------------------------------------------
    def _debounce(self, _=None):
        if self._job:
            self.root.after_cancel(self._job)
        self._job = self.root.after(220, self.run_detection)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("WB Detect", "Could not read that image.")
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.image_path, self.original_img = path, img
        self.result = Result(gray=img, density=rolling_ball(to_density(img, True), 30))
        self.selected = None
        self.multi_sel = set()
        self._detected_backup = []
        # background marks are per-image, so a freshly loaded blot starts clean
        self.bg_regions = []
        self.set_background(img)
        self.redraw()
        self.status.config(text=f"Loaded {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    def run_detection(self):
        if self.original_img is None:
            return
        self.status.config(text="Detecting...")
        self.root.update_idletasks()
        # ROIs you drew or edited by hand survive a re-detection
        keep = [copy.deepcopy(b) for b in getattr(self.result, "bands", [])
                if getattr(b, "manual", False)]
        self.result = detect_bands(self.original_img,
                                   sensitivity=float(self.sens.get()),
                                   bg_radius=int(self.bgr.get()),
                                   dark_bands=bool(self.dark.get()),
                                   deskew=bool(self.deskew.get()),
                                   fill_grid=bool(self.grid.get()))
        if keep:
            kept_boxes = [(b.x, b.x + b.w, b.y, b.y + b.h) for b in keep]
            auto = [b for b in self.result.bands
                    if all(_iou((b.x, b.x + b.w, b.y, b.y + b.h), kb) < 0.30
                           for kb in kept_boxes)]
            self.result.bands = auto + keep
            self.reindex(silent=True)
        removed_bg = self.drop_bands_in_background(redraw=False)
        removed_ai = self.deep_cleanup(silent=True) if self.use_ai.get() else 0
        if removed_bg or removed_ai:
            self._ai_status(f"Removed {removed_bg} ROI(s) inside your background "
                            f"zones and {removed_ai} ROI(s) the AI scored as "
                            "background.")
        self._detected_backup = copy.deepcopy(self.result.bands)
        self.selected = None
        self.multi_sel = set()
        self.set_background(self.result.gray)
        self._clamp_spinboxes()
        self.apply_norm()
        self.redraw()
        self._status_counts()

    def _status_counts(self):
        rows = len({b.row for b in self.result.bands})
        lanes = len({(b.row, b.lane) for b in self.result.bands})
        manual = sum(1 for b in self.result.bands if b.manual)
        self.status.config(
            text=f"{len(self.result.bands)} ROIs ({manual} manual/edited) / {lanes} lanes "
                 f"/ {rows} blot rows | deskew {self.result.angle:+.2f} deg"
                 f"{'  | GPU filtering' if GPU else ''}")

    # ---- canvas / ROI editing --------------------------------------------
    def _size_window(self):
        """Open as large as the screen allows, and try to start maximized."""
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = int(min(1920, sw * 0.96))
        h = int(min(1200, sh * 0.94))
        self.root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 6)}")
        self.root.minsize(1100, 750)
        for state in ("zoomed", "normal"):
            try:
                self.root.state(state)
                break
            except tk.TclError:
                continue

    def _viewport(self) -> tuple[int, int]:
        w = self.canvas.winfo_width() or MAX_PREVIEW_W
        h = self.canvas.winfo_height() or MAX_PREVIEW_H
        return max(50, w - 4), max(50, h - 4)

    def set_background(self, gray: np.ndarray):
        self._disp_gray = gray
        self._render_background()

    def _render_background(self):
        gray = self._disp_gray
        if gray is None:
            return
        h, w = gray.shape[:2]
        vw, vh = self._viewport()
        # "Fit" may now upscale small blots so ROI editing is comfortable.
        self.fit_scale = max(0.05, min(vw / w, vh / h))
        self.scale = self.fit_scale if self.zoom_mode == "fit" else float(self.zoom_mode)
        dw, dh = max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale)))
        interp = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_CUBIC
        disp = cv2.resize(gray, (dw, dh), interpolation=interp) if abs(self.scale - 1.0) > 1e-3 else gray
        self._bg_photo = ImageTk.PhotoImage(Image.fromarray(disp))
        # Center the image in the viewport. When the image is larger than the
        # viewport (zoomed in) the offset collapses to 0 and the scrollbars take
        # over; when it is smaller it snaps to the middle of the canvas.
        iw, ih = self._bg_photo.width(), self._bg_photo.height()
        self._ew, self._eh = max(vw, iw), max(vh, ih)
        self.ox = (self._ew - iw) / 2.0
        self.oy = (self._eh - ih) / 2.0
        self.canvas.config(scrollregion=(0, 0, self._ew, self._eh))
        if hasattr(self, "zoom_lbl"):
            self.zoom_lbl.config(
                text="fit" if self.zoom_mode == "fit" else f"{self.scale * 100:.0f}%")

    def on_canvas_resize(self, _ev=None):
        if self._disp_gray is None:
            return
        if self.zoom_mode == "fit":
            self._render_background()
            self.redraw()

    def set_zoom(self, factor):
        self.zoom_mode = "fit" if factor == "fit" else float(factor)
        self._render_background()
        self.redraw()

    def zoom_fit(self):
        self.set_zoom("fit")

    def zoom_step(self, direction: int):
        cur = self.scale
        steps = list(ZOOM_STEPS)
        if direction > 0:
            nxt = next((z for z in steps if z > cur * 1.01), steps[-1])
        else:
            nxt = next((z for z in reversed(steps) if z < cur * 0.99), steps[0])
        self.set_zoom(nxt)

    def on_wheel(self, ev):
        # The wheel always zooms, anchored on the cursor, so you never need
        # Ctrl/Cmd. Middle-drag pans the canvas.
        delta = ev.delta if getattr(ev, "delta", 0) else (120 if ev.num == 4 else -120)
        ix = self.c2ix(self.canvas.canvasx(ev.x))
        iy = self.c2iy(self.canvas.canvasy(ev.y))
        self.zoom_step(1 if delta > 0 else -1)
        self._anchor_cursor(ix, iy, ev.x, ev.y)
        return "break"

    def _anchor_cursor(self, ix, iy, vx, vy):
        """After a zoom, scroll so the same image pixel stays under the cursor."""
        if self._ew <= 0 or self._eh <= 0:
            return
        wx, wy = self.i2cx(ix), self.i2cy(iy)
        self.canvas.xview_moveto(max(0.0, min(1.0, (wx - vx) / self._ew)))
        self.canvas.yview_moveto(max(0.0, min(1.0, (wy - vy) / self._eh)))

    def _to_canvas(self, ev):
        """Translate widget coords to scrolled canvas coords (in place)."""
        ev.x = self.canvas.canvasx(ev.x)
        ev.y = self.canvas.canvasy(ev.y)
        return ev

    def i2cx(self, v: float) -> float:
        return v * self.scale + self.ox

    def i2cy(self, v: float) -> float:
        return v * self.scale + self.oy

    def c2ix(self, v: float) -> float:
        return (v - self.ox) / self.scale if self.scale else v

    def c2iy(self, v: float) -> float:
        return (v - self.oy) / self.scale if self.scale else v

    def redraw(self):
        self.canvas.delete("all")
        if self._bg_photo is None:
            return
        self.canvas.create_image(self.ox, self.oy, anchor=tk.NW, image=self._bg_photo)
        for (gx0, gx1, gy0, gy1) in self.bg_regions:
            self.canvas.create_rectangle(self.i2cx(gx0), self.i2cy(gy0),
                                         self.i2cx(gx1), self.i2cy(gy1),
                                         outline="#9e9e9e", width=2, dash=(5, 3),
                                         stipple="gray25", fill="#616161")
            self.canvas.create_text(self.i2cx(gx0) + 3, self.i2cy(gy0) + 8,
                                    anchor=tk.W, text="IGNORE", fill="#e0e0e0",
                                    font=("TkDefaultFont", 7, "bold"))
        for b in self.result.bands:
            x0, y0 = self.i2cx(b.x), self.i2cy(b.y)
            x1, y1 = self.i2cx(b.x + b.w), self.i2cy(b.y + b.h)
            base = row_color(b.row)
            if b is self.selected:
                col, wdt = base, 3
            elif b.empty:
                col, wdt = "#9e9e9e", 1
            else:
                col, wdt = base, 2
            is_ref = self._is_ref(b)
            is_ctrl = self._is_ctrl(b)
            if is_ref:
                self.canvas.create_rectangle(x0 - 3, y0 - 3, x1 + 3, y1 + 3,
                                             outline="#ffd54f", width=2, dash=(3, 2))
            elif is_ctrl:
                self.canvas.create_rectangle(x0 - 3, y0 - 3, x1 + 3, y1 + 3,
                                             outline="#7e57c2", width=2, dash=(3, 2))
            if id(b) in self.multi_sel:
                self.canvas.create_rectangle(x0 - 4, y0 - 4, x1 + 4, y1 + 4,
                                             outline="#ff9100", width=2, dash=(2, 2))
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=col, width=wdt,
                                         tags=("roi", f"roi{id(b)}"))
            tag = " REF" if is_ref else (" CTRL" if is_ctrl else "")
            self.canvas.create_text(x0 + 2, max(7, y0 - 6), anchor=tk.W,
                                    text=f"R{b.row}L{b.lane}.{b.index}{tag}",
                                    fill="#ffd54f" if is_ref else base,
                                    font=("TkDefaultFont", 7))
            if b is self.selected:
                for hx, hy in self._handle_points(x0, y0, x1, y1).values():
                    self.canvas.create_rectangle(hx - HANDLE, hy - HANDLE,
                                                 hx + HANDLE, hy + HANDLE,
                                                 outline=base, fill="#1b1b1b")

    @staticmethod
    def _handle_points(x0, y0, x1, y1):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        return {"nw": (x0, y0), "n": (mx, y0), "ne": (x1, y0),
                "e": (x1, my), "se": (x1, y1), "s": (mx, y1),
                "sw": (x0, y1), "w": (x0, my)}

    def _hit_handle(self, b: Band, cx, cy):
        x0, y0 = self.i2cx(b.x), self.i2cy(b.y)
        x1, y1 = self.i2cx(b.x + b.w), self.i2cy(b.y + b.h)
        for name, (hx, hy) in self._handle_points(x0, y0, x1, y1).items():
            if abs(cx - hx) <= HANDLE + 2 and abs(cy - hy) <= HANDLE + 2:
                return name
        return None

    def _hit_band(self, cx, cy) -> Band | None:
        ix, iy = self.c2ix(cx), self.c2iy(cy)
        hits = [b for b in self.result.bands
                if b.x <= ix <= b.x + b.w and b.y <= iy <= b.y + b.h]
        if not hits:
            return None
        return min(hits, key=lambda b: b.w * b.h)

    def on_hover(self, ev):
        ev = self._to_canvas(ev)
        cur = "crosshair"
        if self.tool.get() == "select" and self.selected is not None:
            h = self._hit_handle(self.selected, ev.x, ev.y)
            if h:
                cur = {"nw": "size_nw_se", "se": "size_nw_se", "ne": "size_ne_sw",
                       "sw": "size_ne_sw", "n": "sb_v_double_arrow",
                       "s": "sb_v_double_arrow", "e": "sb_h_double_arrow",
                       "w": "sb_h_double_arrow"}.get(h, "fleur")
            elif self._hit_band(ev.x, ev.y) is self.selected:
                cur = "fleur"
        self.canvas.config(cursor=cur)

    def on_press(self, ev):
        ev = self._to_canvas(ev)
        self.canvas.focus_set()
        if self.result.density is None:
            return
        shift = bool(ev.state & 0x0001)
        if self.tool.get() == "bg":
            self._drag = dict(kind="bg", ox=ev.x, oy=ev.y,
                              rect=self.canvas.create_rectangle(
                                  ev.x, ev.y, ev.x, ev.y, outline="#9e9e9e",
                                  dash=(5, 3), width=2))
            return
        if self.tool.get() == "multi":
            hit = self._hit_band(ev.x, ev.y)
            if hit is not None:
                key = id(hit)
                if key in self.multi_sel:
                    self.multi_sel.discard(key)
                else:
                    self.multi_sel.add(key)
                self.select(hit)
                self.redraw()
                self._update_multi_status()
                return
            self._drag = dict(kind="marquee", ox=ev.x, oy=ev.y, additive=shift,
                              rect=self.canvas.create_rectangle(
                                  ev.x, ev.y, ev.x, ev.y, outline="#ff9100",
                                  dash=(4, 2), width=2))
            return
        draw_mode = self.tool.get() == "draw" or shift

        if not draw_mode and self.selected is not None:
            h = self._hit_handle(self.selected, ev.x, ev.y)
            if h:
                b = self.selected
                self._drag = dict(kind="resize", handle=h, band=b,
                                  box=(b.x, b.y, b.x + b.w, b.y + b.h))
                return

        hit = None if draw_mode else self._hit_band(ev.x, ev.y)
        if hit is not None:
            self.select(hit)
            self._drag = dict(kind="move", band=hit, ox=ev.x, oy=ev.y,
                              box=(hit.x, hit.y, hit.x + hit.w, hit.y + hit.h))
            return

        # start a brand-new ROI
        self._drag = dict(kind="new", ox=ev.x, oy=ev.y,
                          rect=self.canvas.create_rectangle(ev.x, ev.y, ev.x, ev.y,
                                                            outline="#00c853",
                                                            dash=(3, 2), width=2))

    def on_drag(self, ev):
        ev = self._to_canvas(ev)
        d = self._drag
        if not d:
            return
        if d["kind"] in ("new", "bg", "marquee"):
            self.canvas.coords(d["rect"], d["ox"], d["oy"], ev.x, ev.y)
            return

        b = d["band"]
        x0, y0, x1, y1 = d["box"]
        if d["kind"] == "move":
            dx = int(round(self.c2ix(ev.x - d["ox"])))
            dy = int(round(self.c2iy(ev.y - d["oy"])))
            x0, x1 = x0 + dx, x1 + dx
            y0, y1 = y0 + dy, y1 + dy
        else:
            h = d["handle"]
            ix, iy = self.c2ix(ev.x), self.c2iy(ev.y)
            if "n" in h:
                y0 = min(iy, y1 - 2)
            if "s" in h:
                y1 = max(iy, y0 + 2)
            if "w" in h:
                x0 = min(ix, x1 - 2)
            if "e" in h:
                x1 = max(ix, x0 + 2)
        self._set_box(b, x0, y0, x1, y1, remeasure=False)
        self.redraw()

    def on_release(self, ev):
        ev = self._to_canvas(ev)
        d, self._drag = self._drag, None
        if not d:
            return
        if d["kind"] == "bg":
            self.canvas.delete(d["rect"])
            x0, y0 = self.c2ix(min(d["ox"], ev.x)), self.c2iy(min(d["oy"], ev.y))
            x1, y1 = self.c2ix(max(d["ox"], ev.x)), self.c2iy(max(d["oy"], ev.y))
            if (x1 - x0) >= 4 and (y1 - y0) >= 4:
                self.bg_regions.append((int(x0), int(x1), int(y0), int(y1)))
                self.drop_bands_in_background()
                self._ai_status(f"{len(self.bg_regions)} background zone(s) marked - "
                                "they are excluded from detection and used as "
                                "negative examples when you train the AI.")
            self.redraw()
            return
        if d["kind"] == "marquee":
            self.canvas.delete(d["rect"])
            x0, y0 = self.c2ix(min(d["ox"], ev.x)), self.c2iy(min(d["oy"], ev.y))
            x1, y1 = self.c2ix(max(d["ox"], ev.x)), self.c2iy(max(d["oy"], ev.y))
            if (x1 - x0) >= 2 and (y1 - y0) >= 2:
                rect = (x0, x1, y0, y1)
                hit_ids = {id(b) for b in self.result.bands
                          if _rect_overlap((b.x, b.x + b.w, b.y, b.y + b.h), rect) > 0.15}
                if d["additive"]:
                    self.multi_sel |= hit_ids
                else:
                    self.multi_sel = hit_ids
                self._update_multi_status()
            self.redraw()
            return
        if d["kind"] == "new":
            self.canvas.delete(d["rect"])
            x0, y0 = self.c2ix(min(d["ox"], ev.x)), self.c2iy(min(d["oy"], ev.y))
            x1, y1 = self.c2ix(max(d["ox"], ev.x)), self.c2iy(max(d["oy"], ev.y))
            if (x1 - x0) < 3 or (y1 - y0) < 3:
                self.select(None)
                self.redraw()
                return
            self.add_roi(x0, y0, x1, y1)
            return
        b = d["band"]
        self._set_box(b, b.x, b.y, b.x + b.w, b.y + b.h, remeasure=True)
        self.after_edit(b)

    def on_right_click(self, ev):
        ev = self._to_canvas(ev)
        hit = self._hit_band(ev.x, ev.y)
        if hit is not None:
            self.select(hit)
            self.delete_selected()
            return
        ix, iy = self.c2ix(ev.x), self.c2iy(ev.y)
        for r in list(self.bg_regions):
            if r[0] <= ix <= r[1] and r[2] <= iy <= r[3]:
                self.bg_regions.remove(r)
                self.redraw()
                return

    # ---- multi-selection ----------------------------------------------------
    def _update_multi_status(self):
        n = len(self.multi_sel)
        if n:
            self.status.config(text=f"{n} ROI(s) selected - press Delete to remove them.")
        else:
            self._status_counts()

    def delete_any_selected(self):
        """Delete the multi-selected ROIs if any are marked, else the single
        currently-selected ROI."""
        if self.multi_sel:
            before = len(self.result.bands)
            self.result.bands = [b for b in self.result.bands
                                 if id(b) not in self.multi_sel]
            removed = before - len(self.result.bands)
            self.multi_sel.clear()
            self.selected = None
            self.reindex(silent=True)
            self.after_edit(None)
            self.status.config(text=f"Deleted {removed} selected ROI(s).")
        else:
            self.delete_selected()

    # ---- ROI mutation ------------------------------------------------------
    def _set_box(self, b: Band, x0, y0, x1, y1, remeasure=True):
        sig = self.result.density
        if sig is None:
            return
        h, w = sig.shape[:2]
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        x0 = max(0, min(x0, w - 2))
        y0 = max(0, min(y0, h - 2))
        x1 = max(x0 + 2, min(x1, w))
        y1 = max(y0 + 2, min(y1, h))
        if remeasure:
            m = measure_roi(sig, x0, y0, x1, y1)
            for k, v in m.items():
                setattr(b, k, v)
            b.manual = True
        else:
            b.x, b.y, b.w, b.h = x0, y0, x1 - x0, y1 - y0

    def _nearest_row(self, cy: float) -> int:
        if not self.result.bands:
            return 1
        rows = {}
        for b in self.result.bands:
            rows.setdefault(b.row, []).append(b.y + b.h / 2.0)
        best = min(rows, key=lambda r: abs(np.mean(rows[r]) - cy))
        span = max(np.ptp(rows[best]) if len(rows[best]) > 1 else 0, 20)
        return best if abs(np.mean(rows[best]) - cy) <= span else max(rows) + 1

    def _nearest_lane(self, row: int, cx: float) -> int:
        pool = [b for b in self.result.bands if b.row == row] or self.result.bands
        if not pool:
            return 1
        lanes = {}
        for b in pool:
            lanes.setdefault(b.lane, []).append(b.x + b.w / 2.0)
        best = min(lanes, key=lambda l: abs(np.mean(lanes[l]) - cx))
        widths = [b.w for b in pool] or [10]
        if abs(np.mean(lanes[best]) - cx) <= max(np.median(widths), 6):
            return best
        return max(lanes) + 1

    def _next_index(self, row: int, lane: int) -> int:
        same = [b.index for b in self.result.bands if b.row == row and b.lane == lane]
        return (max(same) + 1) if same else 1

    def add_roi(self, x0, y0, x1, y1):
        sig = self.result.density
        if sig is None:
            return
        m = measure_roi(sig, x0, y0, x1, y1)
        row = self._nearest_row(m["y"] + m["h"] / 2.0)
        lane = self._nearest_lane(row, m["x"] + m["w"] / 2.0)
        b = Band(row=row, lane=lane, index=self._next_index(row, lane),
                 manual=True, **m)
        self.result.bands.append(b)
        self.reindex(silent=True)
        self.select(b)
        self.after_edit(b)

    def duplicate_selected(self):
        b = self.selected
        if b is None:
            return
        self.add_roi(b.x + b.w + 2, b.y, b.x + 2 * b.w + 2, b.y + b.h)

    def delete_selected(self):
        b = self.selected
        if b is None:
            return
        self.result.bands = [x for x in self.result.bands if x is not b]
        self.multi_sel.discard(id(b))
        self.selected = None
        self.reindex(silent=True)
        self.after_edit(None)

    def nudge(self, dx, dy):
        b = self.selected
        if b is None:
            return
        self._set_box(b, b.x + dx, b.y + dy, b.x + b.w + dx, b.y + b.h + dy, remeasure=True)
        self.after_edit(b)

    def reindex(self, silent: bool = False):
        """Renumber band indices top-to-bottom inside every (row, lane)."""
        groups: dict[tuple[int, int], list[Band]] = {}
        for b in self.result.bands:
            groups.setdefault((b.row, b.lane), []).append(b)
        for g in groups.values():
            for i, b in enumerate(sorted(g, key=lambda x: x.y), start=1):
                b.index = i
        if not silent:
            self.after_edit(self.selected)

    def revert_edits(self):
        if not self._detected_backup:
            return
        self.result.bands = copy.deepcopy(self._detected_backup)
        self.selected = None
        self.multi_sel = set()
        self.after_edit(None)

    def apply_selected_coords(self):
        b = self.selected
        if b is None:
            return
        b.row = max(1, int(self.sel_row.get()))
        b.lane = max(1, int(self.sel_lane.get()))
        b.index = max(1, int(self.sel_band.get()))
        b.manual = True
        self.after_edit(b)

    def select(self, b: Band | None):
        self.selected = b
        if b is not None:
            self.sel_row.set(b.row)
            self.sel_lane.set(b.lane)
            self.sel_band.set(b.index)

    def after_edit(self, b: Band | None):
        self._clamp_spinboxes()
        self.apply_norm()
        self.redraw()
        self._status_counts()
        if b is not None:
            self._select_in_table(b)

    def _select_in_table(self, b: Band):
        iid = f"{b.row}-{b.lane}-{b.index}-{id(b)}"
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)

    def on_table_select(self, _=None):
        sel = self.tree.selection()
        if not sel:
            return
        target = self.tree.item(sel[0], "tags")
        if not target:
            return
        for b in self.result.bands:
            if str(id(b)) == target[0]:
                self.select(b)
                self.redraw()
                break

    def on_table_click(self, ev):
        """The intensity-profile popup opens ONLY from an explicit click on
        the dedicated 📈 icon column - never from selecting a row."""
        if self.tree.identify_region(ev.x, ev.y) != "cell":
            return
        if self.tree.identify_column(ev.x) != self._plot_col_id:
            return
        row_iid = self.tree.identify_row(ev.y)
        if not row_iid:
            return
        tags = self.tree.item(row_iid, "tags")
        if not tags:
            return
        for b in self.result.bands:
            if str(id(b)) == tags[0]:
                self.select(b)
                self.redraw()
                self.show_lane_profile(b)
                break

    # ---- intensity profile ("hill" plot, like Fiji's Plot Lanes) ----------
    def _lane_profile_for_band(self, b: Band):
        """Vertical intensity profile through the band's lane, wide enough to
        show the peak shape (the 'hill') and some flanking background."""
        sig = self.result.density
        if sig is None:
            return None
        h, w = sig.shape[:2]
        x0, x1 = max(0, int(b.x)), min(w, int(b.x + b.w))
        if x1 <= x0:
            return None
        pad = max(20, int(b.h * 3))
        y0 = max(0, int(b.y) - pad)
        y1 = min(h, int(b.y + b.h) + pad)
        if y1 <= y0:
            return None
        strip = sig[y0:y1, x0:x1]
        prof = strip.mean(axis=1) if strip.size else np.array([0.0])
        prof = gaussian_filter1d(prof.astype(np.float64), 1.0)
        ys = np.arange(y0, y1)
        return ys, prof, (int(b.y), int(b.y + b.h))

    def show_lane_profile(self, b: Band | None):
        if b is None:
            return
        data = self._lane_profile_for_band(b)
        if data is None:
            return
        if self.profile_win is None or not self.profile_win.winfo_exists():
            self.profile_win = tk.Toplevel(self.root)
            self.profile_win.title("Lane intensity profile")
            self.profile_win.geometry("480x300")
            self.profile_canvas = tk.Canvas(self.profile_win, background="#ffffff",
                                            highlightthickness=0)
            self.profile_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
            self.profile_canvas.bind("<Configure>", lambda e: self._redraw_profile())
        else:
            self.profile_win.lift()
        self._profile_data = (*data, b)
        self._redraw_profile()

    def _redraw_profile(self):
        if not self._profile_data or self.profile_canvas is None:
            return
        ys, prof, band_range, b = self._profile_data
        title = (f"R{b.row} L{b.lane} B{b.index}   AUC={b.auc:,.0f}   "
                 f"net={b.net_volume:,.0f}   peak={b.peak:.1f}   SNR={b.snr:.1f}"
                 f"{'  (below SNR)' if b.empty else ''}")
        self._draw_profile(self.profile_canvas, ys, prof, band_range, title)

    @staticmethod
    def _draw_profile(canvas: tk.Canvas, ys: np.ndarray, prof: np.ndarray,
                      band_range: tuple[int, int], title: str = ""):
        canvas.delete("all")
        W = int(canvas.winfo_width() or 460)
        H = int(canvas.winfo_height() or 280)
        mx, my_top, my_bot = 50, 34, 26
        pw = max(40, W - mx - 16)
        ph = max(40, H - my_top - my_bot)
        n = len(prof)
        if n < 2:
            return
        pmin, pmax = float(np.min(prof)), float(np.max(prof))
        rng = max(pmax - pmin, 1e-6)

        def X(i):
            return mx + pw * i / (n - 1)

        def Y(v):
            return my_top + ph * (1 - (v - pmin) / rng)

        b0, b1 = band_range
        inside_idx = [i for i, y in enumerate(ys) if b0 <= y < b1]
        outside_vals = [prof[i] for i in range(n) if i not in set(inside_idx)]
        baseline = float(np.median(outside_vals)) if outside_vals else pmin

        # title
        canvas.create_text(mx, 14, anchor="w", text=title,
                           font=("TkDefaultFont", 9, "bold"), fill="#242824")

        # axes
        canvas.create_line(mx, my_top, mx, my_top + ph, fill="#999")
        canvas.create_line(mx, my_top + ph, mx + pw, my_top + ph, fill="#999")
        canvas.create_text(mx - 6, my_top, anchor="e", text=f"{pmax:.0f}",
                           font=("TkDefaultFont", 7))
        canvas.create_text(mx - 6, my_top + ph, anchor="e", text=f"{pmin:.0f}",
                           font=("TkDefaultFont", 7))
        canvas.create_text(mx, my_top + ph + 12, anchor="w",
                           text=f"y={int(ys[0])}", font=("TkDefaultFont", 7),
                           fill="#777")
        canvas.create_text(mx + pw, my_top + ph + 12, anchor="e",
                           text=f"y={int(ys[-1])}", font=("TkDefaultFont", 7),
                           fill="#777")

        # baseline (dashed red)
        by = Y(baseline)
        canvas.create_line(mx, by, mx + pw, by, fill="#c62828", dash=(4, 2))

        # shaded net-signal area = the region the quantifier calls "the band"
        if inside_idx:
            i0, i1 = min(inside_idx), max(inside_idx)
            area_pts = [(X(i0), by)]
            for i in range(i0, i1 + 1):
                area_pts.append((X(i), Y(prof[i])))
            area_pts.append((X(i1), by))
            flat = [c for p in area_pts for c in p]
            canvas.create_polygon(*flat, fill="#ffe0b2", outline="")
            # band boundary verticals
            canvas.create_line(X(i0), my_top, X(i0), my_top + ph,
                               fill="#ef6c00", dash=(3, 2))
            canvas.create_line(X(i1), my_top, X(i1), my_top + ph,
                               fill="#ef6c00", dash=(3, 2))

        # the intensity curve itself
        pts = [c for i in range(n) for c in (X(i), Y(prof[i]))]
        canvas.create_line(*pts, fill="#1565c0", width=2, smooth=True)

        canvas.create_text(mx + pw, my_top - 14, anchor="e",
                           text="shaded = quantified band (AUC), dashed red = local background",
                           font=("TkDefaultFont", 7), fill="#777")

    # ---- normalization ----------------------------------------------------
    def _clamp_spinboxes(self):
        b = self.result.bands
        if not b:
            return
        max_row = max(x.row for x in b)
        max_lane = max(x.lane for x in b)
        max_band = max(x.index for x in b)
        for var, hi in ((self.v_ref_row, max_row), (self.v_ctrl_row, max_row),
                        (self.v_ref_lane, max_lane), (self.v_ctrl_lane, max_lane),
                        (self.v_ref_band, max_band), (self.v_ctrl_band, max_band)):
            if var.get() > hi:
                var.set(hi)
            if var.get() < 1:
                var.set(1)

    def config(self) -> NormConfig:
        return NormConfig(
            mode=self.mode.get(),
            ref_row=int(self.v_ref_row.get()),
            ref_lane=int(self.v_ref_lane.get()),
            ref_band=int(self.v_ref_band.get()),
            ctrl_row=int(self.v_ctrl_row.get()),
            ctrl_band=int(self.v_ctrl_band.get()),
            ctrl_lane=int(self.v_ctrl_lane.get()),
            exclude_empty=bool(self.excl.get()),
        )

    def ref_strongest(self):
        if not self.result.bands:
            return
        b = max(self.result.bands, key=lambda x: x.auc)
        self.v_ref_row.set(b.row)
        self.v_ref_lane.set(b.lane)
        self.v_ref_band.set(b.index)
        self.mode.set("% of reference band")
        self.apply_norm()

    def ref_from_selected(self):
        """Use the ROI selected on the canvas / in the table as the reference."""
        b = self.selected
        if b is None:
            self.status.config(text="Select a band (click a ROI or a table row) "
                                    "before setting it as reference.")
            return
        self.v_ref_row.set(b.row)
        self.v_ref_lane.set(b.lane)
        self.v_ref_band.set(b.index)
        if self.mode.get() in ("AUC / integrated density (raw)", MODES[0]):
            self.mode.set("% of reference band")
        self.apply_norm()
        self.redraw()
        self.status.config(text=f"Reference band = R{b.row} L{b.lane} B{b.index}"
                                f"  (AUC {b.auc:,.0f})")

    def ctrl_from_selected(self):
        """Use the selected ROI as the loading-control / control-lane anchor."""
        b = self.selected
        if b is None:
            self.status.config(text="Select a band first, then set it as control.")
            return
        self.v_ctrl_row.set(b.row)
        self.v_ctrl_band.set(b.index)
        self.v_ctrl_lane.set(b.lane)
        self.apply_norm()
        self.redraw()
        self.status.config(text=f"Loading control = R{b.row} L{b.lane} B{b.index}")

    def _is_ref(self, b) -> bool:
        return (b.row == int(self.v_ref_row.get())
                and b.lane == int(self.v_ref_lane.get())
                and b.index == int(self.v_ref_band.get()))

    def _is_ctrl(self, b) -> bool:
        return (b.row == int(self.v_ctrl_row.get())
                and b.index == int(self.v_ctrl_band.get()))

    def ref_lane1(self):
        self.v_ref_lane.set(1)
        self.v_ctrl_lane.set(1)
        self.apply_norm()

    def apply_norm(self):
        if not self.result.bands:
            self.tree.delete(*self.tree.get_children())
            return
        cfg = self.config()
        nz = Normalizer(self.result.bands, cfg)
        self.norm = nz.compute()
        self.unit = nz.unit()
        self.norm_hint.config(text=self._hint(cfg))
        self.fill_table()

    def _hint(self, c: NormConfig) -> str:
        m = c.mode
        if m == "% of reference band":
            return (f"Every band is expressed as a percentage of band "
                    f"R{c.ref_row}/L{c.ref_lane}/B{c.ref_band}. "
                    f"Example: band 2 of lane 1 vs band 2 of lane 2 -> set Ref lane 2, Ref band 2.")
        if m == "% of reference lane (same band #)":
            return (f"Each band is divided by the SAME band number in lane {c.ref_lane} "
                    f"of its own blot row.")
        if m == "Loading control ratio (target / control row)":
            return (f"Each band / band {c.ctrl_band} of the same lane in row {c.ctrl_row} "
                    f"(the housekeeping / loading-control row).")
        if m == "Fold change vs control lane":
            return (f"Loading-control ratio, then scaled so lane {c.ctrl_lane} = 1.00. "
                    f"This is the number you put in the figure.")
        if m == "% of lane total":
            return "Each band as a share of all signal detected in its lane."
        if m == "% of row total":
            return "Each band as a share of all signal in its blot row."
        if m == "Relative to row mean":
            return "Each band divided by the mean net volume of its blot row."
        if m == "Same lane across rows (row / reference row)":
            return (f"Every band is divided by the same lane and band number in "
                    f"row {c.ref_row}. Example: row 2 / lane 2 vs row 1 / lane 2 "
                    f"-> set the reference row to 1 (click a band in row 1, press R).")
        if m == "Total-protein normalized":
            return (f"Each band / total signal of the same lane in row {c.ctrl_row} "
                    f"(stain-free / Ponceau), then scaled so lane {c.ctrl_lane} = 1.00.")
        return ("Raw ramp-corrected AUC / integrated density (the shaded 'hill' area "
                "under the lane intensity profile), no normalization applied.")

    def fill_table(self):
        self.tree.delete(*self.tree.get_children())
        unit = getattr(self, "unit", "a.u.")
        bands = sorted(self.result.bands, key=lambda x: (x.row, x.lane, x.index))
        lanes = len({(b.row, b.lane) for b in bands})
        self.table_summary.config(text=f"RESULTS   {len(bands)} BANDS  ·  {lanes} LANES")
        for row_index, b in enumerate(bands):
            val, den, desc = self.norm.get(b.key, (b.auc, 1.0, "none"))
            if unit == "%":
                txt = f"{val:.1f}%"
            elif unit == "x":
                txt = f"{val:.3f}"
            else:
                txt = f"{val:,.0f}"
            self.tree.insert("", tk.END, iid=f"{b.row}-{b.lane}-{b.index}-{id(b)}",
                             tags=(str(id(b)),
                                   f"blotrow{((max(1, b.row) - 1) % len(ROW_COLORS)) + 1}",
                                   "manual" if b.manual else "auto",
                                   *(("refband",) if self._is_ref(b)
                                     else ("ctrlband",) if self._is_ctrl(b) else ())),
                             values=(
                                 b.row, b.lane,
                                 f"{b.net_volume:,.0f}", f"{b.auc:,.0f}",
                                 f"{b.peak:.1f}", f"{b.snr:.1f}",
                                 txt, desc, "📈"))

    def sort_by(self, col):
        if col == "plot":
            return
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]

        def key(v):
            s = v[0].replace(",", "").replace("%", "").strip()
            try:
                return (0, float(s))
            except ValueError:
                return (1, s)

        descending = self._sort_desc
        items.sort(key=key, reverse=descending)
        for name, title in self._heads.items():
            self.tree.heading(name, text=title)
        self.tree.heading(col, text=f"{self._heads[col]} {'▼' if descending else '▲'}")
        self._sort_desc = not descending
        for i, (_, k) in enumerate(items):
            self.tree.move(k, "", i)

    # ---- unified AI (ROI-geometry model + deep classifier) -----------------
    def _combined_ai_summary(self) -> str:
        return ROI_MODEL.summary() + "\n" + DEEP_NET.summary()

    def _ai_status(self, text: str | None = None):
        self.ai_lbl.config(text=text or self._combined_ai_summary())
        self.root.update_idletasks()

    def toggle_ai_panel(self):
        """Fold/unfold the AI-assisted detection panel."""
        if self.ai_body.winfo_ismapped():
            self.ai_body.pack_forget()
            self.ai_toggle_btn.config(text="\u25b6 AI-assisted detection")
        else:
            self.ai_body.pack(side=tk.TOP, fill=tk.X)
            self.ai_toggle_btn.config(text="\u25bc AI-assisted detection")

    def toggle_use_ai(self):
        """Single switch: turns the ROI-fit model on/off, and turns the deep
        classifier on/off too whenever it has actually been trained."""
        on = bool(self.use_ai.get())
        ROI_MODEL.enabled = on
        ROI_MODEL.save()
        if DEEP_NET.ready():
            DEEP_NET.enabled = on
            DEEP_NET.save()
        self._ai_status()
        if self.original_img is not None:
            self.run_detection()

    def _current_boxes(self):
        return [(b.x, b.x + b.w, b.y, b.y + b.h) for b in self.result.bands
                if b.w > 1 and b.h > 1]

    def add_training_example(self):
        if self.result.gray is None or not self.result.bands:
            messagebox.showinfo("AI training",
                                "Load a blot and get the ROIs right first, "
                                "then add it as a training example.")
            return
        boxes = self._current_boxes()
        save_training_sample(self.result.gray, boxes,
                             int(self.bgr.get()), bool(self.dark.get()),
                             bg_regions=list(self.bg_regions))
        n = len(list_training_samples())
        self._ai_status(f"Saved {len(boxes)} ROIs as training example #{n}. "
                        f"Add a few blots, then press Train AI.")

    def train_ai(self):
        """One button trains both halves of the AI on the same saved blots:
        the ROI-fit model (fast, geometry) and the deep classifier (slower,
        band-vs-background). The deep classifier's cfg["use_gpu"] flag comes
        straight from the 'Use GPU (CUDA)' checkbox, so if a CUDA device is
        present and checked, this runs the matmuls on the GPU via CuPy.

        The actual training work happens on a background thread so the GUI
        (and the OS message loop) stay responsive; progress messages are
        posted to a queue and picked up by _poll_train_queue() on the main
        thread via root.after().
        """
        if not list_training_samples():
            messagebox.showinfo("AI training",
                                "No training examples yet.\n\nCorrect the ROIs on a blot "
                                "(and use 'Mark background (ignore)' for areas that must "
                                "never be counted), press 'Add training example', repeat "
                                "for a few blots, then train.")
            return
        if self._train_thread is not None and self._train_thread.is_alive():
            messagebox.showinfo("AI training", "A training run is already in progress.")
            return
        cfg = self._deep_cfg()
        est = cfg["layers"] * cfg["width"] * cfg["epochs"]
        if est > 40_000_000 and not cfg["use_gpu"] and not messagebox.askyesno(
                "AI training",
                f"{cfg['layers']} layers x {cfg['width']} neurons x {cfg['epochs']} "
                "epochs on CPU is a very big run and may take several minutes.\n\n"
                "Start it? (Tip: enable 'Use GPU (CUDA)' if you have a supported "
                "GPU, for a large speedup.)"):
            return

        self.train_btn.config(state="disabled")
        self._ai_status("Training started in the background - the window will "
                        "stay responsive while this runs ...")

        def qprogress(msg):
            # Runs on the WORKER thread - never touch Tk widgets directly here,
            # just hand the message off to the main thread via the queue.
            self._train_queue.put(("status", msg))

        def worker():
            try:
                ok1, msg1 = train_roi_model(ROI_MODEL, progress=qprogress)
                ok2, msg2 = train_deep_net(DEEP_NET, cfg, progress=qprogress)
                self._train_queue.put(("done", (ok1, msg1, ok2, msg2)))
            except Exception as exc:  # keep the worker from dying silently
                self._train_queue.put(("error", str(exc)))

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()
        self.root.after(100, self._poll_train_queue)

    def _poll_train_queue(self):
        """Runs on the MAIN thread via root.after(); safe to touch widgets."""
        try:
            while True:
                kind, payload = self._train_queue.get_nowait()
                if kind == "status":
                    self.ai_lbl.config(text=payload)
                elif kind == "done":
                    ok1, msg1, ok2, msg2 = payload
                    self.train_btn.config(state="normal")
                    self.use_ai.set(ROI_MODEL.enabled or DEEP_NET.enabled)
                    self._ai_status()
                    messagebox.showinfo("AI training",
                                        f"ROI fit - {msg1}\n\nDeep AI - {msg2}")
                    if (ok1 or ok2) and self.original_img is not None:
                        self.run_detection()
                    return  # training finished - stop polling
                elif kind == "error":
                    self.train_btn.config(state="normal")
                    self._ai_status()
                    messagebox.showerror("AI training", f"Training failed:\n{payload}")
                    return
        except queue.Empty:
            pass
        if self._train_thread is not None and self._train_thread.is_alive():
            self.root.after(100, self._poll_train_queue)
        else:
            # thread died without posting "done"/"error" - re-enable the UI
            self.train_btn.config(state="normal")

    def reset_ai(self):
        if not messagebox.askyesno("AI training",
                                   "Delete all saved training examples and reset the AI?"):
            return
        clear_training_samples()
        ROI_MODEL.reset()
        DEEP_NET.reset()
        self.use_ai.set(False)
        self._ai_status()

    def autofit_selected(self):
        b = self.selected
        sig = self.result.density
        if b is None or sig is None:
            return
        h, w = sig.shape[:2]
        mx, my = max(6, int(b.w * 1.2)), max(6, int(b.h * 1.2))
        x0, x1, y0, y1 = tighten_box(sig, b.x, b.x + b.w, b.y, b.y + b.h,
                                     xlim=(max(0, b.x - mx), min(w, b.x + b.w + mx)),
                                     ylim=(max(0, b.y - my), min(h, b.y + b.h + my)))
        self._set_box(b, x0, y0, x1, y1, remeasure=True)
        self.after_edit(b)

    def autofit_all(self):
        sig = self.result.density
        if sig is None or not self.result.bands:
            return
        h, w = sig.shape[:2]
        for b in self.result.bands:
            mx, my = max(6, int(b.w * 1.2)), max(6, int(b.h * 1.2))
            x0, x1, y0, y1 = tighten_box(sig, b.x, b.x + b.w, b.y, b.y + b.h,
                                         xlim=(max(0, b.x - mx), min(w, b.x + b.w + mx)),
                                         ylim=(max(0, b.y - my), min(h, b.y + b.h + my)))
            self._set_box(b, x0, y0, x1, y1, remeasure=True)
        self.after_edit(None)

    # ---- background zones ---------------------------------------------------
    def clear_bg_regions(self):
        self.bg_regions = []
        self._ai_status("Background zones cleared.")
        self.redraw()

    def drop_bands_in_background(self, redraw: bool = True) -> int:
        """Delete every ROI that sits mostly inside a painted background zone."""
        if not self.bg_regions or not self.result.bands:
            return 0
        keep = []
        for b in self.result.bands:
            box = (b.x, b.x + b.w, b.y, b.y + b.h)
            if any(_rect_overlap(box, r) > 0.5 for r in self.bg_regions):
                continue
            keep.append(b)
        n = len(self.result.bands) - len(keep)
        if n:
            self.result.bands = keep
            self.selected = None
            self.reindex(silent=True)
            self.apply_norm()
            if redraw:
                self.redraw()
                self._status_counts()
        return n

    def _deep_cfg(self) -> dict:
        return {"layers": max(1, int(self.deep_layers.get())),
                "width": max(8, int(self.deep_width.get())),
                "epochs": max(1, int(self.deep_epochs.get())),
                "bits": int(self.deep_bits.get()),
                "lr": float(self.deep_lr.get()),
                "threshold": float(self.deep_thr.get()),
                "use_gpu": bool(self.deep_gpu.get())}

    def deep_cleanup(self, silent: bool = False) -> int:
        """Remove every ROI the network scores as background."""
        if not DEEP_NET.ready():
            if not silent:
                messagebox.showinfo("Deep AI", "Train the deep AI first.")
            return 0
        sig = self.result.density
        if sig is None or not self.result.bands:
            return 0
        thr = float(self.deep_thr.get())
        keep, dropped = [], 0
        for b in self.result.bands:
            if getattr(b, "manual", False):
                keep.append(b)
                continue
            score = DEEP_NET.score_box(sig, (b.x, b.x + b.w, b.y, b.y + b.h))
            if score < thr:
                dropped += 1
            else:
                keep.append(b)
        if dropped:
            self.result.bands = keep
            self.selected = None
            self.reindex(silent=True)
            self.apply_norm()
        if not silent:
            self.redraw()
            self._status_counts()
            self._ai_status(f"AI removed {dropped} background-looking ROI(s) "
                            f"(threshold {thr:.2f}).")
        return dropped

    # ---- export -----------------------------------------------------------
    def export_excel(self):
        if not self.result.bands:
            return
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Alignment, Font
            from openpyxl.utils import get_column_letter
        except ImportError:
            messagebox.showerror(
                "Excel export",
                "The openpyxl package is required.\nInstall it with:  pip install openpyxl")
            return

        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel workbook", "*.xlsx")])
        if not path:
            return

        cfg = self.config()
        nz = Normalizer(self.result.bands, cfg)
        # Full set of result types (every normalization mode) per band, so the
        # sheet can be used directly for statistics without re-running the
        # tool for each normalization choice.
        all_norms = nz.compute_all()

        wb = Workbook()
        ws = wb.active
        ws.title = "Results"

        for row in ([f"WB Detect  normalization = {cfg.mode}"],
                    [f"Reference band = R{cfg.ref_row} L{cfg.ref_lane} B{cfg.ref_band}"],
                    [f"Loading control = R{cfg.ctrl_row} L{cfg.ctrl_lane} "
                     f"B{cfg.ctrl_band}"],
                    [f"Deskew = {self.result.angle:+.2f} deg"],
                    []):
            ws.append(row)
        for r in range(1, 5):
            ws.cell(row=r, column=1).font = Font(name="Arial", bold=(r == 1))

        headers = ["Row", "Lane", "Band", "Source", "X", "Y", "W", "H", "Centre Y",
                   "Sigma", "Raw volume", "Background volume", "Net volume",
                   "Hill AUC", "Peak", "SNR", "Normalized", "Denominator",
                   "Relative to", "Below SNR"] + list(MODES)
        head_row = ws.max_row + 1
        ws.append(headers)
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=head_row, column=c)
            cell.font = Font(name="Arial", bold=True)
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        int_fmt = "#,##0;(#,##0);-"
        dec_fmt = "#,##0.0;(#,##0.0);-"
        pct_like = "#,##0.0;(#,##0.0);-"
        n_fixed = len(headers) - len(MODES)
        for b in sorted(self.result.bands, key=lambda x: (x.row, x.lane, x.index)):
            val, den, desc = self.norm.get(b.key, (b.auc, 1.0, "none"))
            mode_vals = all_norms.get(b.key, {m: 0.0 for m in MODES})
            ws.append([b.row, b.lane, b.index,
                       "manual" if b.manual else "auto",
                       b.x, b.y, b.w, b.h,
                       round(b.centre_y, 2), round(b.sigma, 2),
                       round(b.raw_volume, 1), round(b.bg_volume, 1),
                       round(b.net_volume, 1), round(b.auc, 1),
                       round(b.peak, 2), round(b.snr, 2),
                       round(float(val), 4), round(float(den), 4),
                       desc, "yes" if b.empty else "no"]
                      + [round(float(mode_vals[m]), 4) for m in MODES])
            r = ws.max_row
            for c in range(1, len(headers) + 1):
                ws.cell(row=r, column=c).font = Font(name="Arial")
            for c in (5, 6, 7, 8, 11, 12, 13, 14, 15):
                ws.cell(row=r, column=c).number_format = int_fmt
            for c in (9, 10, 17, 18):
                ws.cell(row=r, column=c).number_format = dec_fmt
            for c in range(n_fixed + 1, len(headers) + 1):
                ws.cell(row=r, column=c).number_format = dec_fmt

        widths = [6, 6, 6, 9, 8, 8, 7, 7, 10, 8, 13, 17, 12, 11, 9, 9, 12, 13, 14, 10] \
            + [16] * len(MODES)
        for i, wdt in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = wdt
        ws.freeze_panes = ws.cell(row=head_row + 1, column=1)
        ws.auto_filter.ref = (f"A{head_row}:"
                              f"{get_column_letter(len(headers))}{ws.max_row}")

        try:
            wb.save(path)
        except OSError as exc:
            messagebox.showerror("Excel export", f"Could not save file:\n{exc}")
            return
        self.status.config(text=f"Saved {path}")


if __name__ == "__main__":
    root = tk.Tk()
    WBDetectGUI(root)
    root.mainloop()