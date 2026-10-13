"""Weather playground: apply the pipeline's synthetic adverse weather to a road scene or your own photo.

    streamlit run app.py

NumPy + OpenCV only, so it also runs in the browser (Pyodide/stlite).
"""
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from awsa.scene import road_scene
from awsa.weather import MIXED_PRESETS, SINGLE, weather

st.set_page_config(page_title="Adverse-weather generator", page_icon="🌫️", layout="wide")
st.title("Synthetic adverse weather for driving images")
st.caption("The generators the detection pipeline trains on. Labels survive every effect, so each clean labelled "
           "image becomes a labelled image in fog, rain, snow, dust, sandstorm, haze, night or glare.")

with st.sidebar:
    src = st.radio("Image", ["Procedural road scene", "Upload a photo"])
    if src.startswith("Upload"):
        up = st.file_uploader("JPG or PNG", type=["jpg", "jpeg", "png"])
        img = np.asarray(Image.open(io.BytesIO(up.getvalue())).convert("RGB")) / 255.0 if up else None
    else:
        img = road_scene(seed=st.number_input("Scene seed", 0, 100, 0))
    if img is not None and max(img.shape[:2]) > 900:  # keep it quick in the browser
        scale = 900 / max(img.shape[:2])
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(
            (int(img.shape[1] * scale), int(img.shape[0] * scale)))) / 255.0
    severity = st.slider("Severity", 0.0, 1.0, 0.5, 0.05)
    st.caption("Depth: 0 at the bottom of the frame (near) rising to the top (far). The pipeline uses "
               "Depth-Anything-V2 instead when it is available.")

tab_one, tab_all, tab_res = st.tabs(["One condition", "All conditions", "Pipeline results"])

if img is None:
    st.info("Upload a photo in the sidebar.")
else:
    with tab_one:
        kind = st.radio("Condition", ["single", "time of day + weather"], horizontal=True)
        name = st.selectbox("Weather", SINGLE if kind == "single" else list(MIXED_PRESETS))
        c1, c2 = st.columns(2)
        c1.image(img, caption="clean", use_container_width=True, clamp=True)
        c2.image(weather(img, name, severity), caption=f"{name}, severity {severity:.2f}",
                 use_container_width=True, clamp=True)
    with tab_all:
        names = list(SINGLE) + list(MIXED_PRESETS)
        cols = st.columns(4)
        for i, n in enumerate(names):
            cols[i % 4].image(weather(img, n, severity), caption=n, use_container_width=True, clamp=True)

with tab_res:
    st.warning("These come from a 20-image, 3-epoch smoke test of the full pipeline, run before the depth-prior "
               "fix (see the README). They show the pipeline runs end to end, not that the method works.")
    m = Path("results/metrics.json")
    if m.exists():
        d = json.loads(m.read_text())
        st.table([{"cycle": c["cycle"], "action": c["action"], "model": c["variant"], "mean AP@0.5": c["mean_ap"]}
                  for c in d["cycles"]])
    for f, cap in (("results/failure_map_baseline.png", "AP by weather × severity, baseline"),
                   ("results/failure_map_final.png", "after adaptation"),
                   ("results/self_improvement_curve.png", "mean AP per cycle")):
        if Path(f).exists():
            st.image(f, caption=cap)
