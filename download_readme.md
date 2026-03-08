# `download.py` Guide

This document explains how `download.py` builds cached training data for the Waymo Scenario-Gen workflow.
It is written for a new collaborator who needs to run, debug, and trust the pipeline without reading the full source first.

## What This Pipeline Does

`download.py` converts raw Waymo TFRecord scenarios into PyTorch shard files used by training.

Default source paths (Before, get access from https://waymo.com/open/licensing/):
- train: `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/`
- val: `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation/`

Pipeline steps:
1. **GCS list**: list TFRecord files from Waymo GCS train/val prefixes.
2. **Batch download**: download a small batch of TFRecord files into a temporary local folder.
3. **Scenario parse**: stream scenarios and parse protobufs into trajectory/map arrays.
4. **Sample build**: generate model-ready samples with history, neighbors, map context, static features, masks, and targets.
5. **Shard + manifest write**: flush samples into `samples_*.pt` shards and write `manifest.pt` metadata.

## Prerequisites

Required tools/libraries:
- Python 3.10+
- `gsutil` authenticated for GCS access
- TensorFlow
- PyTorch
- `waymo-open-dataset` (protos + sim agents utilities) - Require Linux (I managed to run on WSL2)
- NumPy

Expected local paths created/used by defaults:
- `./waymo_batch_temp`
- `./waymo_cache_v2/train`
- `./waymo_cache_v2/val`

Important safety note:
- `REBUILD_EXISTING_CACHE=True` by default, so existing `train` and `val` cache directories are deleted and recreated at startup.

## Default Configuration (From Code)

### Data scope

| Constant | Default | Meaning |
|---|---:|---|
| `MAX_TRAIN_SCENARIOS` | `5000` | Maximum train scenarios to process. |
| `MAX_VAL_SCENARIOS` | `1000` | Maximum validation scenarios to process. |
| `FILES_PER_BATCH` | `1` | Number of TFRecord files downloaded per streaming batch. |
| `TEST_MODE` | `False` | If `True`, stream only one source file per split for quick checks. |

### Temporal layout and sampling

| Constant | Default | Meaning |
|---|---:|---|
| `PAST_STEPS` | `10` | Historical steps before current index. |
| `CURRENT_TIME_INDEX` | `10` | Anchor index for the canonical present. |
| `HISTORY_STEPS` | `11` | `CURRENT_TIME_INDEX + 1`. |
| `FUTURE_STEPS` | `80` | Future target horizon. |
| `TOTAL_SCENE_STEPS` | `91` | `HISTORY_STEPS + FUTURE_STEPS`. |
| `NEIGHBORS_K` | `6` | Number of nearest neighbors retained. |
| `ANCHOR_MIN_T` | `10` | First anchor timestep. |
| `ANCHOR_MAX_T` | `70` | Last auxiliary anchor timestep considered. |
| `ANCHOR_STRIDE` | `5` | Anchor spacing when `MULTI_ANCHOR=True`. |
| `MIN_FUTURE_VALID_AUX` | `20` | Minimum valid future steps for non-primary anchors. |
| `MULTI_ANCHOR` | `True` | Generate samples from anchor times `10, 15, ..., 70`. |

### Hard-case balancing

| Constant | Default | Meaning |
|---|---:|---|
| `HIGH_SPEED_STEP_THRESHOLD` | `2.6` | Threshold on per-step speed magnitude for high-speed duplication. |
| `EDGE_NEAR_THRESHOLD_M` | `6.0` | Distance-to-road-edge threshold for near-edge duplication. |
| `HARD_CASE_DUPLICATION_FACTOR` | `2` | Duplicate count for hard cases (high-speed or near-edge). |

### Storage and performance

| Constant | Default | Meaning |
|---|---:|---|
| `SEED` | `2026` | Seed used for Python/NumPy/PyTorch randomness setup. |
| `NUM_WORKERS` | `10` | Multiprocessing workers for scenario processing. |
| `SHARD_SIZE` | `12000` | Buffered samples before shard flush (train). |
| `val_shard_size` | `max(1, SHARD_SIZE // 2)` | Validation uses half the train shard size (with floor at 1). |
| `LOCAL_STORAGE_BUDGET_GB` | `100.0` | Total local budget assumption. |
| `RESERVED_FREE_SPACE_GB` | `15.0` | Space intentionally left unused. |
| `TEMP_DOWNLOAD_BUDGET_GB` | `2.0` | Max size of temporary download folder. |
| `TRAIN_CACHE_BUDGET_GB` | `65.0` | Train cache stop budget. |
| `VAL_CACHE_BUDGET_GB` | `8.0` | Validation cache stop budget. |
| `REBUILD_EXISTING_CACHE` | `True` | Recreate cache directories before writing. |

## Sample Semantics and Tensor Shapes

All dynamic motion features are expressed in an **anchor-local coordinate frame**:
- anchor position is local origin
- anchor heading defines local x-axis
- headings are represented by `sin`/`cos`

### Per-sample tensors

- `hist`: `[H, 13]` where `H=11`
- `nbr`: `[K, 10]` where `K=6`
- `map`: `[5]`
- `static`: `[7]`
- `target`: `[F, 4]` where `F=80`
- `masks`: `hist_valid [H]`, `target_valid [F]`, `nbr_valid [K]`, `map_valid [1]`
- `meta`: `[7]`

### Feature definitions

`hist[h] = [rel_x, rel_y, rel_vx, rel_vy, sin(dh), cos(dh), 1.0, type_one_hot(3), length, width, height]`

`nbr[k] = [rel_x, rel_y, rel_vx, rel_vy, sin(dh_ij), cos(dh_ij), nbr_type_one_hot(3), 1.0]`

`map = [dist_nearest_lane, lane_rel_sin, lane_rel_cos, dist_nearest_road_edge, map_valid]`

`static = [type_one_hot(3), length, width, height, anchor_speed]`

`target[f] = [future_rel_x, future_rel_y, sin(future_dh), cos(future_dh)]`

### Sampling rules that affect dataset size

- Primary anchor (`t=10`) requires at least **1** valid future step.
- Auxiliary anchors (`t>10`) require at least **20** valid future steps (`MIN_FUTURE_VALID_AUX`).
- `nbr` uses nearest-by-distance neighbors at anchor time and pads to `K=6`.
- Hard cases are duplicated when speed or edge-nearness criteria are met.
- Masks are stored as **float** masks (`0.0/1.0`), not booleans.

## Output Contract

### Shard files

Pattern:
- `samples_00000.pt`, `samples_00001.pt`, ...

Each shard is a `dict` with keys:
- `hist`, `nbr`, `map`, `static`, `target`, `meta`
- `masks` (nested dict)

`masks` keys:
- `hist_valid`, `target_valid`, `nbr_valid`, `map_valid`

### Manifest file

Each split (`train`, `val`) writes `manifest.pt` with:
- `max_scenarios`: configured cap for that split
- `scenarios_seen`: scenarios actually consumed
- `total_samples`: total samples written
- `H`, `F`, `K`: tensor horizon/config values
- `anchor_stride`, `min_future_valid`, `multi_anchor`
- `high_speed_step_threshold`, `edge_near_threshold_m`, `hard_case_duplication_factor`
- `anchor_counts`: sample count by anchor timestep
- `shards`: sorted list of shard file paths
- `cache_size_gb`: final cache size
- `cache_budget_gb`: configured budget
- `stopped_for_budget`: `True` when stopped due to size cap

## Public Function Contracts

### `build_training_samples_from_scenario(...) -> dict`

Returns one sample dictionary batch for a scenario with keys:
- `hist`: `[N, H, 13]`
- `nbr`: `[N, K, 10]`
- `map`: `[N, 5]`
- `static`: `[N, 7]`
- `target`: `[N, F, 4]`
- `masks.hist_valid`: `[N, H]`
- `masks.target_valid`: `[N, F]`
- `masks.nbr_valid`: `[N, K]`
- `masks.map_valid`: `[N, 1]`
- `meta`: `[N, 7]`

If no valid samples exist for a scenario, it returns empty tensors with `N=0` and the same trailing dimensions.

### `write_sample_shards(...) -> list[str]`

Consumes streamed scenarios, parallelizes sample building, flushes shard files, writes `manifest.pt`, and returns sorted shard paths.

Side effects:
- writes `samples_*.pt`
- writes `manifest.pt`
- may stop early when `cache_size >= max_cache_gb`

### `validate_cache_split(...) -> None`

Assertion-based validation of one split:
- shard presence
- shape checks for `hist`, `target`, `nbr`
- finite-value checks on all tensors and masks
- anchor coverage check when `multi_anchor=True`

Raises assertion errors on invalid cache artifacts.

## Validation and Failure Modes

What validation checks:
- at least one shard exists
- expected dimensions for `H/F/K`
- no non-finite values in data or masks
- no missing anchor bins for expected anchor sequence (when multi-anchor enabled)

Budget behavior:
- if cache directory already exceeds budget, split returns existing shards immediately
- during writing, split stops once `cache_size_gb` reaches budget
- this is recorded in manifest as `stopped_for_budget=True`

Common failures and what they mean:
- `gsutil` command failure: missing auth, wrong project permissions, or bad GCS path
- temp-budget `RuntimeError`: `waymo_batch_temp` exceeded `TEMP_DOWNLOAD_BUDGET_GB`
- anchor coverage assertion: multi-anchor sampling produced zero samples for one or more expected anchors
- no shards assertion: scenario filtering or upstream download issues prevented sample production

## Quick Run + Verify

### Run

```bash
python download.py
```

If using a specific env executable:

```bash
python download.py
```

### Expected terminal signals

During execution you should see:
- framework/device prints (`TensorFlow`, `PyTorch`, `Device`)
- split discovery logs (`[TRAIN] files found: ...`, `[VAL] files found: ...`)
- first-sample sanity print:
  - `=== First Sample Validation ===`
  - `hist shape: ...`
  - `target shape: ...`
- periodic shard logs:
  - `Wrote shard ... (cache size: ... GB / ... GB)`
- final split summaries:
  - `Shard build complete for train ...`
  - `Shard build complete for val ...`
- validation success:
  - `Validated waymo_cache_v2/train successfully.`
  - `Validated waymo_cache_v2/val successfully.`
- final completion:
  - `All processing completed successfully.`

### Post-run checks

Inspect:
- `./waymo_cache_v2/train/samples_*.pt`
- `./waymo_cache_v2/val/samples_*.pt`
- `./waymo_cache_v2/train/manifest.pt`
- `./waymo_cache_v2/val/manifest.pt`

For anchor distribution and stop reason, read each split manifest keys:
- `anchor_counts`
- `stopped_for_budget`
