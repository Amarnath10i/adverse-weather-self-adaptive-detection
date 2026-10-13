"""A procedurally drawn road scene, so the demo needs no third-party photos."""
import cv2
import numpy as np


def road_scene(h=360, w=640, seed=0):
    """RGB float image in [0, 1]: sky, trees, road with lane markings, a few cars and a street light."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    horizon = int(h * 0.45)
    sky = np.linspace(0, 1, horizon)[:, None, None]
    img[:horizon] = (1 - sky) * np.array([0.45, 0.65, 0.95]) + sky * np.array([0.85, 0.90, 0.97])
    img[horizon:] = np.array([0.32, 0.55, 0.25])  # grass
    for x in rng.integers(0, w, 9):  # trees on the horizon
        r = int(rng.integers(18, 38))
        cv2.circle(img, (int(x), horizon - r // 2), r, (0.15, 0.38, 0.15), -1)
    road = np.array([[w * 0.47, horizon], [w * 0.53, horizon], [w * 0.98, h], [w * 0.02, h]], np.int32)
    cv2.fillPoly(img, [road], (0.28, 0.28, 0.30))
    for i in range(8):  # dashed centre line, growing toward the camera
        y0 = horizon + (h - horizon) * (i / 8) ** 1.6
        y1 = horizon + (h - horizon) * ((i + 0.5) / 8) ** 1.6
        cv2.line(img, (w // 2, int(y0)), (w // 2, int(y1)), (0.95, 0.95, 0.85), max(1, int(1 + 6 * i / 8)))
    for depth, dx, colour in ((0.25, -0.08, (0.75, 0.1, 0.1)), (0.55, 0.12, (0.15, 0.25, 0.7)),
                              (0.85, -0.22, (0.9, 0.9, 0.9))):
        y = int(horizon + (h - horizon) * depth)
        cw, ch = int(30 + 170 * depth), int(18 + 90 * depth)
        cx = int(w / 2 + dx * w * depth * 1.6)
        cv2.rectangle(img, (cx - cw // 2, y - ch), (cx + cw // 2, y), colour, -1)
        cv2.rectangle(img, (cx - cw // 3, y - ch), (cx + cw // 3, y - int(ch * 0.55)), (0.55, 0.7, 0.8), -1)
        for sx in (-1, 1):
            cv2.circle(img, (cx + sx * cw // 3, y), max(2, ch // 6), (0.05, 0.05, 0.05), -1)
    cv2.line(img, (int(w * 0.9), horizon - 60), (int(w * 0.9), h - 40), (0.4, 0.4, 0.4), 4)
    cv2.circle(img, (int(w * 0.87), horizon - 60), 9, (1.0, 0.97, 0.85), -1)  # lamp (a glare source)
    return np.clip(img, 0, 1)
