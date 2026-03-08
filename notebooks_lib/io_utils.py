"""I/O helpers shared by EDA and inference notebooks.

These utilities wrap TFRecord iteration and GCS file management so notebooks can
focus on analysis steps instead of boilerplate shell calls.
"""

from __future__ import annotations

import math
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

GCS_BASE = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"
GCS_VAL_PATH = f"{GCS_BASE}/validation/"
LOCAL_WAYMO_CACHE_DIR = "./waymo_val_cache"
VAL_FILE_CACHE: list[str] | None = None


def run_cmd(cmd: list[str]) -> str:
    """Run a shell command and return stdout text."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def list_tfrecords(gcs_prefix: str) -> list[str]:
    """List and sort TFRecord files under a GCS prefix."""
    output = run_cmd(["gsutil", "ls", gcs_prefix])
    files = [line.strip() for line in output.splitlines() if line.strip()]
    return sorted(files)


def download_single_tfrecord(gcs_uri: str, local_dir: Path, force: bool = False) -> Path:
    """Download one TFRecord file locally unless already present."""
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / Path(gcs_uri).name
    if destination.exists() and not force:
        print(f"Using existing file: {destination}")
        return destination

    run_cmd(["gsutil", "cp", gcs_uri, str(destination)])
    print(f"Downloaded: {gcs_uri} -> {destination}")
    return destination


def iter_scenarios(tfrecord_path: Path, max_scenarios: int | None = None):
    """Yield parsed `Scenario` protos from a TFRecord file."""
    dataset = tf.data.TFRecordDataset([str(tfrecord_path)])
    for idx, raw in enumerate(dataset.as_numpy_iterator()):
        if max_scenarios is not None and idx >= max_scenarios:
            break
        yield scenario_pb2.Scenario.FromString(raw)


def collect_split_stats(tfrecord_path: Path, split_name: str, max_scenarios: int | None = None) -> dict:
    """Compute high-level dataset stats for one TFRecord split sample."""
    tracks_per_scenario = []
    map_features_per_scenario = []
    dyn_states_per_scenario = []
    valid_ratio_per_scenario = []
    mean_speed_per_scenario = []

    object_type_counter = Counter()
    example_scenario = None

    for scenario in iter_scenarios(tfrecord_path, max_scenarios=max_scenarios):
        if example_scenario is None:
            example_scenario = scenario

        tracks_per_scenario.append(len(scenario.tracks))
        map_features_per_scenario.append(len(scenario.map_features))
        dyn_states_per_scenario.append(len(scenario.dynamic_map_states))

        total_states = 0
        valid_states = 0
        speeds = []

        for track in scenario.tracks:
            object_type_counter[int(track.object_type)] += 1
            for state in track.states:
                total_states += 1
                if state.valid:
                    valid_states += 1
                    speed = math.hypot(state.velocity_x, state.velocity_y)
                    if math.isfinite(speed):
                        speeds.append(speed)

        valid_ratio_per_scenario.append(valid_states / total_states if total_states > 0 else np.nan)
        mean_speed_per_scenario.append(float(np.mean(speeds)) if speeds else np.nan)

    return {
        "split": split_name,
        "scenario_count": len(tracks_per_scenario),
        "tracks_per_scenario": np.array(tracks_per_scenario, dtype=np.float32),
        "map_features_per_scenario": np.array(map_features_per_scenario, dtype=np.float32),
        "dyn_states_per_scenario": np.array(dyn_states_per_scenario, dtype=np.float32),
        "valid_ratio_per_scenario": np.array(valid_ratio_per_scenario, dtype=np.float32),
        "mean_speed_per_scenario": np.array(mean_speed_per_scenario, dtype=np.float32),
        "object_type_counter": object_type_counter,
        "example_scenario": example_scenario,
    }


def local_cache_path_for_gcs_file(gcs_path: str, cache_dir: str = LOCAL_WAYMO_CACHE_DIR) -> str:
    """Resolve local cache path for a remote TFRecord object."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, os.path.basename(gcs_path))


def ensure_local_tfrecord(
    gcs_path: str,
    cache_dir: str = LOCAL_WAYMO_CACHE_DIR,
    force_refresh: bool = False,
    verbose: bool = True,
) -> str:
    """Ensure a GCS TFRecord exists in local cache and return local path."""
    local_path = local_cache_path_for_gcs_file(gcs_path, cache_dir=cache_dir)
    if force_refresh and os.path.exists(local_path):
        os.remove(local_path)

    if not os.path.exists(local_path):
        if verbose:
            print(f"Downloading {gcs_path} -> {local_path}")
        subprocess.run(["gsutil", "cp", gcs_path, local_path], check=True)
    elif verbose:
        print(f"Using cached TFRecord: {local_path}")

    return local_path


def list_validation_tfrecords(gcs_val_path: str = GCS_VAL_PATH, refresh: bool = False) -> list[str]:
    """List validation TFRecords from GCS with local-cache fallback.

    If `gsutil ls` is unavailable (offline/auth-limited environments), this
    falls back to already-downloaded validation TFRecords under
    `LOCAL_WAYMO_CACHE_DIR`.
    """
    global VAL_FILE_CACHE
    if VAL_FILE_CACHE is not None and not refresh:
        return VAL_FILE_CACHE
    try:
        result = subprocess.run(["gsutil", "ls", gcs_val_path], capture_output=True, text=True, check=True)
        VAL_FILE_CACHE = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return VAL_FILE_CACHE
    except subprocess.CalledProcessError:
        cache_dir = Path(LOCAL_WAYMO_CACHE_DIR)
        if cache_dir.exists():
            local_files = sorted(str(p) for p in cache_dir.glob("validation.tfrecord-*"))
            if local_files:
                VAL_FILE_CACHE = local_files
                return VAL_FILE_CACHE
        raise


def sample_cross_file_scenarios(
    n: int = 10,
    seed: int = 2026,
    cleanup_local: bool = False,
    cache_dir: str = LOCAL_WAYMO_CACHE_DIR,
    force_refresh: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Reservoir-sample one scenario per randomly selected validation file."""
    val_files = list_validation_tfrecords()
    if len(val_files) == 0:
        raise RuntimeError("No validation files found in GCS validation path.")

    n = min(int(n), len(val_files))
    rng = np.random.default_rng(seed)
    chosen_file_indices = sorted(int(x) for x in rng.choice(len(val_files), size=n, replace=False))

    sampled = []
    for file_index in chosen_file_indices:
        target_file = val_files[file_index]
        local_temp = ensure_local_tfrecord(
            target_file,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            verbose=verbose,
        )

        dataset = tf.data.TFRecordDataset([local_temp])
        chosen_bytes = None
        chosen_idx = -1
        count = 0
        for idx, bytes_example in enumerate(dataset.as_numpy_iterator()):
            count += 1
            if rng.random() < (1.0 / count):
                chosen_bytes = bytes_example
                chosen_idx = idx

        if chosen_bytes is None:
            if cleanup_local and os.path.exists(local_temp):
                os.remove(local_temp)
            raise RuntimeError(f"No scenarios found in file: {target_file}")

        scenario = scenario_pb2.Scenario.FromString(chosen_bytes)
        sampled.append(
            {
                "file_index": int(file_index),
                "file_path": target_file,
                "local_tfrecord": local_temp,
                "scenario_index": int(chosen_idx),
                "scenario_count_in_file": int(count),
                "scenario_id": str(scenario.scenario_id),
                "scenario_proto": scenario,
            }
        )

        if cleanup_local and os.path.exists(local_temp):
            os.remove(local_temp)

    unique_pairs = {(item["file_index"], item["scenario_index"]) for item in sampled}
    assert len(unique_pairs) == len(sampled), "Duplicate (file_index, scenario_index) sampled"
    return sampled
