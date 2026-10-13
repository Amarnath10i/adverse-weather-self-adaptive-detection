"""Physically motivated synthetic weather for driving images (extracted from the notebook).

Every function takes an RGB image (uint8 or float in [0, 1]) and returns a
float image in [0, 1]. Fog, mist, dust, sand and snow thicken with distance
through a depth map (0 = near, 1 = far); without one, a vertical prior is
used (top of the frame is far). Geometry is unchanged (or only
micro-perturbed by heat shimmer), so bounding-box labels stay valid.
"""
import cv2
import numpy as np


def _as_float_rgb(img):
    """Accept uint8/float, HxWx3, return RGB float [0,1]."""
    img = np.asarray(img).astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)

def _default_depth(H, W):
    """Vertical prior: top of frame = far (used when no depth map is given)."""
    return np.tile(np.linspace(1.0, 0.2, H)[:, None], (1, W)).astype(np.float32)

def add_fog(clean, depth01, beta=0.5, atmos=0.85, max_depth=6.0):
    """Koschmieder atmospheric scattering: I = J*t + A*(1-t), t = exp(-beta*depth)."""
    clean = _as_float_rgb(clean)
    d = np.clip(depth01, 0, 1).astype(np.float32) * max_depth
    t = np.clip(np.exp(-beta * d), 0.05, 1.0)[..., None]
    return np.clip(clean * t + np.float32(atmos) * (1 - t), 0, 1)

def add_mist(clean, severity=0.5, depth01=None, seed=0, max_depth=6.0):
    """Mist: a gentler, low-lying fog (thicker toward the ground) + slight scatter blur.
       Cooler-white veil, weaker depth falloff and a higher transmission floor than fog."""
    clean = _as_float_rgb(clean); H, W = clean.shape[:2]
    if depth01 is None:
        depth01 = _default_depth(H, W)
    d = np.clip(depth01, 0, 1).astype(np.float32) * max_depth
    beta = 0.12 + severity * 0.45
    t = np.clip(np.exp(-beta * d), 0.30, 1.0)                     # higher floor -> gentler
    ground = np.linspace(0.5, 1.0, H)[:, None].astype(np.float32)  # denser near the ground
    veil = ((1.0 - t) * ground)[..., None]
    atmos = np.array([0.90, 0.92, 0.94], np.float32)              # slightly cool white
    out = clean * (1 - veil) + atmos * veil
    out = cv2.GaussianBlur(out, (0, 0), 0.8 + severity * 1.2)     # moisture scatter
    return np.clip(out, 0, 1)

def add_rain(clean, intensity=0.5, angle=75, streak_len=25, seed=0):
    clean = _as_float_rgb(clean); H, W = clean.shape[:2]
    rng = np.random.default_rng(seed)
    layer = np.zeros((H, W), np.float32)
    n = int(intensity * 12000)
    ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
    layer[ys, xs] = rng.uniform(0.5, 1.0, n).astype(np.float32)
    k = max(3, int(streak_len))
    kernel = np.zeros((k, k), np.float32)
    cv2.line(kernel, (k // 2, 0), (k // 2, k - 1), 1.0, 1)
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle - 90, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum(); kernel = kernel / s if s > 1e-6 else kernel
    streaks = np.clip(cv2.filter2D(layer, -1, kernel), 0, 1)
    out = clean * (1.0 - intensity * 0.2) + np.stack([streaks * 0.8] * 3, -1)
    return np.clip(out, 0, 1)

def add_night(clean, severity=0.5, seed=0):
    clean = _as_float_rgb(clean); rng = np.random.default_rng(seed)
    gamma = 1.0 + severity * 3.0
    out = np.power(np.clip(clean, 1e-6, 1.0), gamma)       # clip BEFORE power (fix)
    out[..., 0] *= (1 - severity * 0.30)                   # R down
    out[..., 1] *= (1 - severity * 0.15)                   # G down
    out[..., 2] *= (1 + severity * 0.10)                   # B up (cool cast)
    out = out + rng.normal(0, severity * 0.06, clean.shape).astype(np.float32)
    bright = (clean.mean(-1) > 0.8).astype(np.float32)
    bloom = cv2.GaussianBlur(bright, (0, 0), 12)
    out = out + bloom[..., None] * 0.25 * severity
    return np.clip(out, 0, 1)

def add_glare(clean, severity=0.5, light_xy=None):
    clean = _as_float_rgb(clean); H, W = clean.shape[:2]
    if light_xy is None:
        g = clean.mean(-1); y, x = np.unravel_index(g.argmax(), g.shape); light_xy = (int(x), int(y))
    Y, X = np.mgrid[0:H, 0:W]
    dist = np.sqrt((X - light_xy[0]) ** 2 + (Y - light_xy[1]) ** 2)
    glow = np.exp(-dist / (80 + severity * 220))
    bloom = glow * severity * 0.6
    bloom3 = np.stack([bloom, bloom * 0.95, bloom * 0.85], -1)   # warm
    return np.clip(clean + bloom3 + severity * 0.15, 0, 1)

def add_dust(clean, severity=0.5, depth01=None, seed=0, max_depth=6.0):
    """Dust haze: warm ochre depth-aware veil + low-freq turbulence + desaturation."""
    clean = _as_float_rgb(clean); H, W = clean.shape[:2]
    rng = np.random.default_rng(seed)
    if depth01 is None:
        depth01 = _default_depth(H, W)
    d = np.clip(depth01, 0, 1) * max_depth
    beta = 0.35 + severity * 0.9
    t = np.clip(np.exp(-beta * d), 0.05, 1.0)[..., None]
    turb = rng.random((H // 16 + 1, W // 16 + 1)).astype(np.float32)
    turb = cv2.GaussianBlur(cv2.resize(turb, (W, H)), (0, 0), 8)
    turb = 0.85 + 0.3 * (turb - turb.mean())
    A = np.array([0.78, 0.62, 0.38], np.float32) * turb[..., None]   # RGB ochre
    out = clean * t + A * (1 - t)
    gray = out.mean(-1, keepdims=True)
    out = out * (1 - 0.25 * severity) + gray * (0.25 * severity)
    return np.clip(out, 0, 1)

def add_sand(clean, severity=0.5, depth01=None, seed=0, max_depth=6.0):
    """Sandstorm: heavier/darker warm veil than dust + wind-blown horizontal grit streaks
       + strong desaturation + slight blur. (A more severe sibling of `dust`.)"""
    clean = _as_float_rgb(clean); H, W = clean.shape[:2]
    rng = np.random.default_rng(seed)
    if depth01 is None:
        depth01 = _default_depth(H, W)
    d = np.clip(depth01, 0, 1) * max_depth
    beta = 0.6 + severity * 1.4                                  # denser than dust
    t = np.clip(np.exp(-beta * d), 0.03, 1.0)[..., None]
    turb = rng.random((H // 16 + 1, W // 16 + 1)).astype(np.float32)
    turb = cv2.GaussianBlur(cv2.resize(turb, (W, H)), (0, 0), 10)
    turb = 0.8 + 0.4 * (turb - turb.mean())
    A = np.array([0.82, 0.60, 0.32], np.float32) * turb[..., None]   # darker ochre
    out = clean * t + A * (1 - t)
    grit = np.zeros((H, W), np.float32)                          # wind-blown particulate
    n = int(severity * 9000)
    ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
    grit[ys, xs] = rng.uniform(0.3, 0.8, n).astype(np.float32)
    k = 15; mk = np.zeros((k, k), np.float32)
    cv2.line(mk, (0, k // 2), (k - 1, k // 2), 1.0, 1); mk /= mk.sum()   # horizontal blur
    grit = cv2.filter2D(grit, -1, mk)
    out = out + np.stack([grit * 0.5, grit * 0.38, grit * 0.2], -1) * severity
    gray = out.mean(-1, keepdims=True)
    out = out * (1 - 0.40 * severity) + gray * (0.40 * severity)
    out = cv2.GaussianBlur(out, (0, 0), 0.6 + severity * 1.0)
    return np.clip(out, 0, 1)

def add_humid(clean, severity=0.5, seed=0):
    """High humidity / heat haze: warm low-contrast veil, contrast compression,
       heat-shimmer displacement (stronger lower in frame) and a faint moist bloom.
       Geometry is only micro-perturbed, so bounding-box labels stay valid."""
    clean = _as_float_rgb(clean); H, W = clean.shape[:2]
    rng = np.random.default_rng(seed)
    haze = np.array([0.80, 0.80, 0.78], np.float32)              # warm-grey moisture veil
    a = 0.12 + severity * 0.35
    out = clean * (1 - a) + haze * a
    m = out.mean()                                              # compress contrast to mid-grey
    out = m + (out - m) * (1 - 0.25 * severity)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)              # heat shimmer (denser low)
    amp = severity * 2.5 * (yy / max(1, H - 1))
    shift = amp * np.sin(2 * np.pi * (yy / 18.0) + rng.uniform(0, 2 * np.pi))
    out = cv2.remap(out, (xx + shift).astype(np.float32), yy,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    bright = (out.mean(-1) > 0.7).astype(np.float32)            # moist-air bloom
    bloom = cv2.GaussianBlur(bright, (0, 0), 8)
    out = out + bloom[..., None] * 0.12 * severity
    return np.clip(out, 0, 1)

# severity-parameterised dispatch (single "severity dial" in [0,1] per weather)
WEATHER_FNS = {
    "fog":   lambda img, s, depth: add_fog(img, depth, beta=0.2 + s * 1.3),
    "mist":  lambda img, s, depth: add_mist(img, severity=s, depth01=depth),
    "rain":  lambda img, s, depth: add_rain(img, intensity=s),
    "snow":  lambda img, s, depth: add_snow(img, severity=s, depth01=depth),
    "dust":  lambda img, s, depth: add_dust(img, severity=s, depth01=depth),
    "sand":  lambda img, s, depth: add_sand(img, severity=s, depth01=depth),
    "humid": lambda img, s, depth: add_humid(img, severity=s),
    "night": lambda img, s, depth: add_night(img, severity=s),
    "glare": lambda img, s, depth: add_glare(img, severity=s),
}

def add_snow(clean, severity=0.5, depth01=None, seed=0, max_depth=6.0):
    """Snow: cool bright depth-aware veil + two flake layers + slight downward streak."""
    clean = _as_float_rgb(clean); H, W = clean.shape[:2]
    rng = np.random.default_rng(seed)
    if depth01 is None:
        depth01 = _default_depth(H, W)
    d = np.clip(depth01, 0, 1) * max_depth
    t = np.clip(np.exp(-(0.2 + severity * 0.7) * d), 0.15, 1.0)[..., None]
    base = clean * t + np.array([0.90, 0.92, 0.96], np.float32) * (1 - t)
    flakes = np.zeros((H, W), np.float32)
    for scale, count, blur in [(3, int(severity * 1200), 1.2), (2, int(severity * 2500), 0.6)]:
        n = max(0, count)
        ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
        f = np.zeros((H, W), np.float32); f[ys, xs] = rng.uniform(0.6, 1.0, n).astype(np.float32)
        f = cv2.GaussianBlur(cv2.dilate(f, np.ones((scale, scale), np.uint8)), (0, 0), blur)
        flakes = np.maximum(flakes, f)
    k = 7; mk = np.zeros((k, k), np.float32); cv2.line(mk, (k // 2, 0), (k // 2, k - 1), 1.0, 1); mk /= mk.sum()
    flakes = np.maximum(flakes, cv2.filter2D(flakes, -1, mk) * 0.6)
    return np.clip(base + np.stack([flakes] * 3, -1) * 0.9, 0, 1)

def apply_weather(img_rgb01, weather, severity, depth01):
    return WEATHER_FNS[weather](img_rgb01, float(severity), depth01)


# == 1b - Mixed weather x time-of-day ========================================
# Compose a time-of-day re-lighting with one or more weather effects so the
# pipeline also sees realistic *co-occurring* conditions (night+rain, dusk+
# sandstorm, dawn+mist, ...). Order: relight -> weather scattering. Labels are
# unchanged (global photometric ops), so mixed images are labelled for free.

def add_time_of_day(clean, tod, severity=1.0, seed=0):
    """Re-light a clean [0,1] RGB image for a time of day. Applied BEFORE weather
       so scattering sits on the correct ambient light. severity=0 -> untouched."""
    clean = _as_float_rgb(clean)
    s = float(np.clip(severity, 0.0, 1.0))
    if tod == "night":
        return add_night(clean, severity=0.5 + 0.5 * s, seed=seed)
    #        gain, (R,G,B) white-balance, contrast
    GRADES = {
        "dawn":      (0.82, (1.05, 0.99, 1.12), 0.96),  # cool, dim
        "morning":   (0.96, (1.03, 1.00, 0.99), 1.00),  # soft warm
        "noon":      (1.10, (1.00, 1.00, 1.00), 1.08),  # bright neutral
        "afternoon": (0.98, (1.07, 1.00, 0.93), 1.03),  # golden hour
        "dusk":      (0.74, (1.12, 0.92, 0.88), 0.95),  # warm-orange, dim
    }
    gain, wb, contrast = GRADES.get(tod, (1.0, (1.0, 1.0, 1.0), 1.0))
    gain     = 1.0 + (gain - 1.0) * s
    contrast = 1.0 + (contrast - 1.0) * s
    wb = np.array([1.0 + (c - 1.0) * s for c in wb], np.float32)
    out = clean * gain * wb
    out = 0.5 + (out - 0.5) * contrast
    return np.clip(out, 0.0, 1.0)

# realistic day-cycle x weather presets: (time_of_day, [weathers to stack])
MIXED_PRESETS = {
    "dawn_mist":      ("dawn",      ["mist"]),
    "morning_fog":    ("morning",   ["fog"]),
    "noon_glare":     ("noon",      ["glare"]),
    "noon_humid":     ("noon",      ["humid"]),
    "afternoon_dust": ("afternoon", ["dust"]),
    "dusk_rain":      ("dusk",      ["rain"]),
    "dusk_sandstorm": ("dusk",      ["sand"]),
    "night_rain":     ("night",     ["rain"]),
    "night_fog":      ("night",     ["fog"]),
    "night_snow":     ("night",     ["snow"]),
}

def apply_mixed(img_rgb01, preset, severity, depth01):
    """Time-of-day relight followed by the preset's stacked weather effects."""
    tod, ws = MIXED_PRESETS[preset]
    out = add_time_of_day(img_rgb01, tod, severity=severity)
    for w in ws:
        out = apply_weather(out, w, severity, depth01)
    return out

for _p in MIXED_PRESETS:
    WEATHER_FNS[_p] = (lambda p: lambda img, s, depth: apply_mixed(img, p, s, depth))(_p)

SINGLE = ("fog", "mist", "rain", "snow", "dust", "sand", "humid", "night", "glare")


def weather(img, name, severity, depth01=None):
    """Apply one condition (or a mixed preset) at a severity in [0, 1]; depth defaults to the vertical prior."""
    img = _as_float_rgb(img)
    if depth01 is None:
        depth01 = _default_depth(*img.shape[:2])
    return WEATHER_FNS[name](img, float(severity), depth01)
