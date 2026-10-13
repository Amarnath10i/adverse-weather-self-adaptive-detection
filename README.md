# Adverse-Weather Self-Adaptive Object Detection

A research pipeline that trains a driving-scene object detector to be **robust across
every common adverse condition** — fog, mist, rain, snow, dust, sandstorm, high humidity /
haze, night and glare — using **physics-based synthetic weather generation**, a
**self-improving / adaptive training loop**, and **self-pruning + capacity-growth** driven
by the model's own weights and outputs.

> **Status: work in progress.** The pipeline runs end to end in a smoke-test configuration
> (20 source images, 3 baseline epochs). The full run (`AWSA_FULL=1`) has not been done yet,
> so there are no results yet that show the method works; see [Results](#results).

---

## Try the weather generator

```bash
pip install -r requirements.txt
streamlit run app.py
```

The playground applies every condition and the time-of-day presets (dawn mist, dusk sandstorm, night rain…) to a procedurally drawn road scene or your own photo, at any severity. It needs only NumPy and OpenCV, so it also runs in the browser. The generators live in [`awsa/weather.py`](awsa/weather.py), extracted from the notebook's first section.

**Bug fixed while building it:** the fallback depth prior was upside down. It made the bottom of the frame, the road right in front of the camera, the *farthest* point, so fog, dust and sand were thickest nearest the car. It now runs from near at the bottom to far at the top, in the notebook too. The pipeline prefers Depth-Anything-V2 depth, but the committed smoke-test results below fell back to the prior, so they predate this fix.

## What it does

| Stage | Idea |
|-------|------|
| **1. Synthetic weather** | Depth-aware, physically-motivated corruptions applied to clean labelled images. Labels are preserved, so every synthetic image is labelled for free. |
| **2. Depth estimation** | Depth-Anything-V2 (with a gradient fallback) makes fog/mist/dust/sand/snow thicken correctly with distance. |
| **3. Leak-free data** | Clean images are split into train/val/test **first**, then weather is applied per split — no augmentation leakage across splits. |
| **4. Baseline detector** | YOLOv8 trained on the full weather grid. |
| **5. Failure map** | Per-class AP@0.5 swept over `weather × severity` — a heat-map of where the model is weak. |
| **6. Self-adaptation** | Weak `(weather, severity)` cells are **oversampled** and the model is fine-tuned; capacity **grows** (`n → s → m`) when accuracy plateaus. |
| **7. Self-pruning** | **Weight-based** (L1-magnitude) and **output/activation-based** channel pruning, then fine-tune to recover — reported as sparsity vs. mAP. |
| **8. Test-time adaptation** | Optional BatchNorm recalibration for unseen conditions. |
| **9. Real-world eval** | Hook for RTTS / DAWN / ExDark (YOLO format) to test synthetic→real transfer. |

## Adverse conditions modelled

`fog · mist · rain · snow · dust · sand (sandstorm) · humid/haze · night · glare`

Each is a single-parameter *severity dial* in `[0, 1]`, so the whole failure map is a clean
`weather × severity` grid.

## Self-improvement, adaptive behaviour, pruning & block growth

- **Adaptive**: the training distribution is re-weighted each cycle toward the conditions the
  model currently fails on (targeted oversampling), with a clean/easy/hard mix to prevent
  catastrophic forgetting.
- **Self-improvement**: cycles repeat until mean-AP plateaus.
- **Grow blocks (output/weight driven)**: on plateau the network's capacity is escalated by
  variant growth (`yolov8n → s → m`), the skip-connection-safe analogue of "adding blocks".
- **Self-pruning (weight & output driven)**: L1-magnitude prune of every conv (weights) plus
  activation-magnitude channel pruning (outputs), followed by recovery fine-tuning.

## Run the full pipeline

```bash
pip install -r requirements-train.txt

# Scaled run (CPU-friendly, completes end-to-end):
jupyter nbconvert --to notebook --execute --inplace adverse_weather_self_adaptive_kaggle.ipynb

# Full run (GPU recommended):
AWSA_FULL=1 jupyter nbconvert --to notebook --execute --inplace adverse_weather_self_adaptive_kaggle.ipynb
```

### Resumability

Every heavy stage (dataset builds, each training run, each severity sweep, each adaptation
cycle, and pruning) checkpoints to `artifacts/state/` with `.done` markers. If the machine is
shut down mid-run, **re-running resumes from the last finished stage** instead of starting over.
Set `AWSA_RESUME=0` to force a clean run.

### Environment switches

| Variable | Default | Meaning |
|----------|---------|---------|
| `AWSA_FULL` | `0` | `1` = full data / epochs / cycles (GPU) |
| `AWSA_RESUME` | `1` | `0` = ignore saved state, run from scratch |
| `AWSA_ROOT` | `./artifacts` | where datasets / runs / state are written |
| `AWSA_BDD` | *(unset)* | path to a YOLO-format BDD100K dir to use real data instead of COCO128 |

## Results

`results/` holds the output of the **smoke-test run** (COCO128 subset, 20 source images,
YOLOv8n at 384 px, 3 baseline epochs, 2 adaptation cycles, severities 0.3 and 0.7). It shows
the pipeline executes; at this scale the numbers are noise, not evidence:

| cycle | action | model | mean AP@0.5 over the weather grid |
|---|---|---|---|
| 0 | baseline | YOLOv8n | 20.2 |
| 1 | targeted oversampling of weak cells | YOLOv8n | 8.3 |
| 2 | capacity growth | YOLOv8s | 24.0 |
| - | L1 + activation pruning, then fine-tune | YOLOv8s | 0.1 |

Known issues found by this run:
- **Targeted fine-tuning hurt** (20.2 to 8.3). With 20 source images and 2 epochs this may be
  noise or forgetting of the clean/easy mix; the full run has to tell which.
- **Pruning is broken**: it reports 0% conv sparsity yet mean AP collapses to 0.1, so the
  pruning/recovery stage corrupts weights rather than removing channels. It must be fixed
  before any pruning result is reported.

Next steps: fix pruning, then run `AWSA_FULL=1` on BDD100K (`AWSA_BDD`) and evaluate
synthetic-to-real transfer on RTTS / DAWN.
