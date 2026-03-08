# Diffusion Model for Scenario Generation on Waymo Self-driving Motion dataset.

<div align="center">

| Scenario 9 | Scenario 26 | Scenario 10 |
| :---: | :---: | :---: |
| <img src="figures/Scenario 9/fig0_rollout_animation.gif" width="100%" /> | <img src="figures/Scenario 26/fig0_rollout_animation.gif" width="100%" /> | <img src="figures/Scenario 10/fig0_rollout_animation.gif" width="100%" /> |

</div>
This project builds a **Waymo Motion Scenario-Gen training pipeline** from raw TFRecords to cached PyTorch shards, trains a **chunked diffusion trajectory model**, and runs inference with post-processing plus report-ready visualizations.

## Project Goal
Predict 80 future timesteps of agent motion (x, y) using structured scene context:
- ego/agent history
- neighboring agents
- local map geometry
- static object attributes

## Repository Workflow
1. `download.py`
Builds local train/val caches from Waymo GCS TFRecords and writes shard files (`samples_*.pt`) plus manifests.
2. `EDA.ipynb`
Lightweight EDA on one training and one validation TFRecord (no cache build).
3. `train.ipynb`
PyTorch-only training on cached shards, EMA checkpoints, diagnostics, and checkpoint export.
4. `inference.ipynb`
Loads the trained checkpoint, runs rollout inference on validation scenarios, applies post-processing, and exports paper figures.

## Data Pipeline Summary (`download.py`)
This script acts as the primary data ingestion and preprocessing engine. It streams raw records directly from Google Cloud Storage Bucket and compiles them into structured PyTorch shards optimized for training.

Dataset Source and detailed raw fields can be found in https://waymo.com/open/data/motion/

More detailed on `download_readme.md`

## Training Summary (`train.ipynb`)
- Model: `ChunkDiffusionModel` + `ConditionEncoder`
- Diffusion steps: `T=200`, sample steps `50`
- Guidance scale: `1.2`
- Token tables: position `512`, trajectory `1024`, chunk length `5`
- Batch config: batch size `192`, grad accumulation `3`, epochs `10`
- Optimizer: AdamW + mixed precision + EMA

### Main Training Artifacts
- `checkpoints/latest_training.pt`
- `checkpoints/best_ema.pt`
- `chunk_diffusion_pytorch_only.pt`
- logs under `logs/`

## Inference and Reporting (`inference.ipynb`)
The notebook supports scenario rollout, post-process ablations, and figure export in `figures/`.

### Reported Metrics (`figures/report_insights.txt`)
- Primary quality: `step1_l2=0.235 m`, `ADE80=7.693 m`, `FDE80=22.626 m`
- Safety realism (selected mode): `off-road=23`, `collisions=0`, `slip=0.4305`
- Runtime trade-off: fastest mode `legacy (9.589s)`, slowest `bicycle_full (44.602s)`

### Generated Figures
- `fig1_main_overlay.{png,pdf}`
- `fig2_keyframe_strip.{png,pdf}`
- `fig3_error_horizon.{png,pdf}`
- `fig4_safety_bars.{png,pdf}`
- `fig5_runtime_pareto.{png,pdf}`
- `fig6_case_gallery.{png,pdf}`

## Environment Requirements
Use your Waymo environment (example: `waymo_env`) with at least:
- Python 3.10+
- PyTorch
- TensorFlow
- `waymo-open-dataset` - Note: Only on Linux, so use WSL2
- NumPy, Pandas, Matplotlib, tqdm
- `gsutil` (for GCS TFRecord access)

## Quick Start
1. Build cached shards:
```bash
python download.py
```
2. Run exploratory analysis:
- Open and run `EDA.ipynb`
3. Train model and export checkpoint:
- Open and run `train.ipynb`
4. Run inference + generate report figures:
- Open and run `inference.ipynb`
