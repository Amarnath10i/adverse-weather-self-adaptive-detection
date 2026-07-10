# Adverse-Weather Self-Adaptive Object Detection

A research pipeline that trains a driving-scene object detector to be **robust across
every common adverse condition** — fog, mist, rain, snow, dust, sandstorm, high humidity /
haze, night and glare — using **physics-based synthetic weather generation**, a
**self-improving / adaptive training loop**, and **self-pruning + capacity-growth** driven
by the model's own weights and outputs.

> Status: work in progress. Runs end-to-end on CPU in a *scaled* configuration and scales
> to GPU / full data with a single switch (`AWSA_FULL=1`).

---

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

## Run it

```bash
pip install -r requirements.txt

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

See [`results/`](results/) — failure maps, the self-improvement curve, metrics JSON, and the
final / pruned model weights are written there when the notebook runs.

## License

[MIT](LICENSE)
