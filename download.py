import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs

SEED = 2026
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 3

PAST_STEPS = 10
CURRENT_TIME_INDEX = 10
HISTORY_STEPS = CURRENT_TIME_INDEX + 1
FUTURE_STEPS = 80
TOTAL_SCENE_STEPS = HISTORY_STEPS + FUTURE_STEPS
NEIGHBORS_K = 6
ANCHOR_MIN_T = CURRENT_TIME_INDEX
ANCHOR_MAX_T = 70
ANCHOR_STRIDE = 5
MIN_FUTURE_VALID_AUX = 20
MULTI_ANCHOR = True

# Long-tail emphasis inspired by high-speed failure analysis in challenge reports.
HIGH_SPEED_STEP_THRESHOLD = 2.6  # ~26 m/s (approx 58 mph) in 0.1s delta units
EDGE_NEAR_THRESHOLD_M = 6.0
HARD_CASE_DUPLICATION_FACTOR = 2

MAX_TRAIN_SCENARIOS = 5000
MAX_VAL_SCENARIOS = 1000
FILES_PER_BATCH = 1
SHARD_SIZE = 12000
REBUILD_EXISTING_CACHE = True
TEST_MODE = False  
NUM_WORKERS = 10 

LOCAL_STORAGE_BUDGET_GB = 100.0
RESERVED_FREE_SPACE_GB = 15.0
TEMP_DOWNLOAD_BUDGET_GB = 2.0
TRAIN_CACHE_BUDGET_GB = 65.0
VAL_CACHE_BUDGET_GB = 8.0

CACHE_ROOT = Path("./waymo_cache_v2")
TRAIN_CACHE_DIR = CACHE_ROOT / "train"
VAL_CACHE_DIR = CACHE_ROOT / "val"
BATCH_DIR = Path("./waymo_batch_temp")

GCS_BASE = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"
GCS_TRAIN_PATH = f"{GCS_BASE}/training/"
GCS_VAL_PATH = f"{GCS_BASE}/validation/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert HISTORY_STEPS == 11
assert FUTURE_STEPS == 80
assert TOTAL_SCENE_STEPS == 91
assert ANCHOR_MIN_T == CURRENT_TIME_INDEX
assert HARD_CASE_DUPLICATION_FACTOR >= 1
assert TRAIN_CACHE_BUDGET_GB + VAL_CACHE_BUDGET_GB + TEMP_DOWNLOAD_BUDGET_GB <= (LOCAL_STORAGE_BUDGET_GB - RESERVED_FREE_SPACE_GB)

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024 ** 3)

def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total

class StreamingWaymoDataset:
    """
    Streams TFRecord files from Google Cloud Storage in small batches.
    This avoids downloading the entire dataset at once to respect local storage limits.
    """
    def __init__(
        self,
        gcs_path: str,
        max_scenarios: int,
        files_per_batch: int,
        batch_dir: Path,
        split_name: str,
        test_mode: bool = False,
    ):
        self.gcs_path = gcs_path
        self.max_scenarios = max_scenarios
        self.files_per_batch = files_per_batch
        self.batch_dir = Path(batch_dir)
        self.split_name = split_name
        self.test_mode = test_mode

        result = subprocess.run(
            ["gsutil", "ls", gcs_path], capture_output=True, text=True, check=True
        )
        self.all_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        
        if test_mode:
            self.all_files = self.all_files[:1]  
            print(f"[{split_name}] TEST MODE: using only 1 file for validation")
        
        print(f"[{split_name}] files found: {len(self.all_files)}")

    def __iter__(self):
        scenarios_yielded = 0
        file_cursor = 0

        while scenarios_yielded < self.max_scenarios:
            batch = self.all_files[file_cursor:file_cursor + self.files_per_batch]
            if not batch:
                break
            file_cursor += len(batch)

            subprocess.run(
                ["gsutil", "-m", "cp"] + batch + [str(self.batch_dir) + "/"],
                check=True,
                capture_output=True,
            )

            temp_size_gb = bytes_to_gb(directory_size_bytes(self.batch_dir))
            if temp_size_gb > TEMP_DOWNLOAD_BUDGET_GB:
                raise RuntimeError(
                    f"Temporary download directory reached {temp_size_gb:.2f} GB, above TEMP_DOWNLOAD_BUDGET_GB={TEMP_DOWNLOAD_BUDGET_GB:.2f} GB."
                )

            local_files = sorted(
                str(self.batch_dir / fn)
                for fn in os.listdir(self.batch_dir)
                if (self.batch_dir / fn).is_file()
            )
            tf_dataset = tf.data.TFRecordDataset(local_files)

            for bytes_example in tf_dataset.as_numpy_iterator():
                if scenarios_yielded >= self.max_scenarios:
                    break
                yield bytes_example
                scenarios_yielded += 1

            for fp in local_files:
                try:
                    os.remove(fp)
                except OSError:
                    pass

def wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))

def to_local(vec_xy: np.ndarray, anchor_heading: float) -> np.ndarray:
    """
    Converts global X and Y coordinates into the local coordinate frame of the target agent for a given time step.
    The local frame is defined such that the X-axis points in the direction of the agent's heading, and the Y-axis is perpendicular to it.
    """
    c = math.cos(anchor_heading)
    s = math.sin(anchor_heading)
    x, y = vec_xy
    return np.array([c * x + s * y, -s * x + c * y], dtype=np.float32)

def object_type_one_hot(object_type: int) -> np.ndarray:
    if object_type == TYPE_VEHICLE:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if object_type == TYPE_PEDESTRIAN:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if object_type == TYPE_CYCLIST:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def scenario_id_to_int(scenario_id: str) -> int:
    try:
        return int(scenario_id[:16], 16)
    except Exception:
        return abs(hash(scenario_id)) % (2**31 - 1)

def _extract_polyline_points(polyline) -> np.ndarray:
    if polyline is None or len(polyline) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(p.x, p.y) for p in polyline], dtype=np.float32)

def extract_map_arrays(scenario_proto: scenario_pb2.Scenario, point_stride: int = 5) -> dict:
    """
    Parses protobuf map features to extract lane point coordinates and road edge locations.
    Downsamples the points using point_stride to reduce memory usage.
    """
    lane_points = []
    lane_dirs = []
    road_edge_points = []

    for map_feature in scenario_proto.map_features:
        feature_name = map_feature.WhichOneof("feature_data")
        if feature_name not in {"lane", "road_line", "road_edge"}:
            continue

        feature = getattr(map_feature, feature_name)
        polyline = getattr(feature, "polyline", None)
        pts = _extract_polyline_points(polyline)
        if pts.shape[0] == 0:
            continue

        if point_stride > 1:
            pts = pts[::point_stride]
            if pts.shape[0] == 1 and len(polyline) > 1:
                pts = np.array(
                    [(polyline[0].x, polyline[0].y), (polyline[-1].x, polyline[-1].y)],
                    dtype=np.float32,
                )

        if feature_name == "road_edge":
            road_edge_points.append(pts)
            continue

        lane_points.append(pts)
        if pts.shape[0] >= 2:
            seg = pts[1:] - pts[:-1]
            seg_norm = np.linalg.norm(seg, axis=1, keepdims=True) + 1e-6
            unit = seg / seg_norm
            dirs = np.concatenate([unit, unit[-1:]], axis=0)
        else:
            dirs = np.array([[1.0, 0.0]], dtype=np.float32)
        lane_dirs.append(dirs)

    if lane_points:
        lane_points = np.concatenate(lane_points, axis=0)
        lane_dirs = np.concatenate(lane_dirs, axis=0)
    else:
        lane_points = np.zeros((0, 2), dtype=np.float32)
        lane_dirs = np.zeros((0, 2), dtype=np.float32)

    if road_edge_points:
        road_edge_points = np.concatenate(road_edge_points, axis=0)
    else:
        road_edge_points = np.zeros((0, 2), dtype=np.float32)

    return {
        "lane_points": lane_points,
        "lane_dirs": lane_dirs,
        "road_edge_points": road_edge_points,
    }

def compute_map_context(anchor_xy: np.ndarray, anchor_heading: float, map_cache: dict) -> np.ndarray:
    lane_points = map_cache["lane_points"]
    lane_dirs = map_cache["lane_dirs"]
    road_edge_points = map_cache["road_edge_points"]

    dist_lane = 0.0
    lane_sin = 0.0
    lane_cos = 1.0
    dist_edge = 0.0
    map_valid = 0.0

    if lane_points.shape[0] > 0:
        lane_delta = lane_points - anchor_xy[None, :]
        lane_dist2 = np.sum(lane_delta ** 2, axis=1)
        lane_idx = int(np.argmin(lane_dist2))
        dist_lane = float(np.sqrt(lane_dist2[lane_idx]))
        lane_dir = lane_dirs[lane_idx]
        lane_theta = math.atan2(float(lane_dir[1]), float(lane_dir[0]))
        lane_rel = wrap_angle(lane_theta - anchor_heading)
        lane_sin = math.sin(lane_rel)
        lane_cos = math.cos(lane_rel)
        map_valid = 1.0

    if road_edge_points.shape[0] > 0:
        edge_delta = road_edge_points - anchor_xy[None, :]
        edge_dist2 = np.sum(edge_delta ** 2, axis=1)
        dist_edge = float(np.sqrt(np.min(edge_dist2)))
        map_valid = 1.0

    return np.array([dist_lane, lane_sin, lane_cos, dist_edge, map_valid], dtype=np.float32)

def _empty_sample_dict(H: int, F: int, K: int) -> dict:
    return {
        "hist": torch.zeros((0, H, 13), dtype=torch.float32),
        "nbr": torch.zeros((0, K, 10), dtype=torch.float32),
        "map": torch.zeros((0, 5), dtype=torch.float32),
        "static": torch.zeros((0, 7), dtype=torch.float32),
        "target": torch.zeros((0, F, 4), dtype=torch.float32),
        "masks": {
            "hist_valid": torch.zeros((0, H), dtype=torch.float32),
            "target_valid": torch.zeros((0, F), dtype=torch.float32),
            "nbr_valid": torch.zeros((0, K), dtype=torch.float32),
            "map_valid": torch.zeros((0, 1), dtype=torch.float32),
        },
        "meta": torch.zeros((0, 7), dtype=torch.float32),
    }

def build_training_samples_from_scenario(
    scenario_proto: scenario_pb2.Scenario,
    H: int,
    F: int,
    K: int,
    current_time_index: int,
    challenge_type_val: int,
    anchor_stride: int = ANCHOR_STRIDE,
    min_future_valid: int = MIN_FUTURE_VALID_AUX,
    multi_anchor: bool = MULTI_ANCHOR,
    high_speed_step_threshold: float = HIGH_SPEED_STEP_THRESHOLD,
    edge_near_threshold_m: float = EDGE_NEAR_THRESHOLD_M,
    hard_case_duplication_factor: int = HARD_CASE_DUPLICATION_FACTOR,
) -> dict:
    """
    Extracts structured training samples from a single scenario record.
    
    This function performs the main feature engineering work:
    * Loops over multiple time anchors.
    * Builds historical state vectors for the target agent.
    * Finds the closest neighboring agents.
    * Calculates distances to map lanes and road edges.
    * Extracts the future trajectory ground truth.
    """
    full_traj = trajectory_utils.ObjectTrajectories.from_scenario(scenario_proto)
    sim_ids = tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario_proto, challenge_type_val))
    trajectories = full_traj.gather_objects_by_id(sim_ids)

    x = trajectories.x.numpy().astype(np.float32)
    y = trajectories.y.numpy().astype(np.float32)
    z = trajectories.z.numpy().astype(np.float32)
    heading = trajectories.heading.numpy().astype(np.float32)
    valid = trajectories.valid.numpy().astype(bool)

    length = trajectories.length.numpy().astype(np.float32)
    width = trajectories.width.numpy().astype(np.float32)
    height = trajectories.height.numpy().astype(np.float32)

    object_type = trajectories.object_type.numpy().astype(np.int32)
    object_id = trajectories.object_id.numpy().astype(np.int64)

    n_obj, n_steps = x.shape
    if H != (current_time_index + 1):
        raise ValueError(f"Expected H == current_time_index + 1, got H={H} and current_time_index={current_time_index}")
    if n_steps <= current_time_index + 1:
        return _empty_sample_dict(H, F, K)

    vx = np.zeros_like(x)
    vy = np.zeros_like(y)
    vx[:, 1:] = x[:, 1:] - x[:, :-1]
    vy[:, 1:] = y[:, 1:] - y[:, :-1]

    map_cache = extract_map_arrays(scenario_proto)
    scenario_hash = float(scenario_id_to_int(scenario_proto.scenario_id))

    hist_list = []
    nbr_list = []
    map_list = []
    static_list = []
    target_list = []
    hist_valid_list = []
    target_valid_list = []
    nbr_valid_list = []
    map_valid_list = []
    meta_list = []

    max_anchor = min(ANCHOR_MAX_T, n_steps - 2)
    anchor_times = [current_time_index]
    if multi_anchor:
        for t_anchor in range(current_time_index + anchor_stride, max_anchor + 1, anchor_stride):
            anchor_times.append(t_anchor)

    for anchor_t in anchor_times:
        for i in range(n_obj):
            type_oh = object_type_one_hot(int(object_type[i]))

            if not valid[i, anchor_t]:
                continue

            anchor_xy = np.array([x[i, anchor_t], y[i, anchor_t]], dtype=np.float32)
            anchor_heading = float(heading[i, anchor_t])
            anchor_z = float(z[i, anchor_t])
            anchor_v = np.array([vx[i, anchor_t], vy[i, anchor_t]], dtype=np.float32)
            size_vec = np.array([length[i, anchor_t], width[i, anchor_t], height[i, anchor_t]], dtype=np.float32)

            hist_feat = np.zeros((H, 13), dtype=np.float32)
            hist_valid = np.zeros((H,), dtype=np.float32)

            for h_idx, s in enumerate(range(anchor_t - H + 1, anchor_t + 1)):
                if s < 0 or s >= n_steps:
                    continue

                base_feat = np.concatenate([type_oh, size_vec], axis=0)
                if not valid[i, s]:
                    hist_feat[h_idx, 7:] = base_feat
                    continue

                rel_xy = to_local(np.array([x[i, s], y[i, s]], dtype=np.float32) - anchor_xy, anchor_heading)
                rel_v = to_local(np.array([vx[i, s], vy[i, s]], dtype=np.float32), anchor_heading)
                dhead = wrap_angle(float(heading[i, s]) - anchor_heading)

                hist_feat[h_idx] = np.array(
                    [
                        rel_xy[0],
                        rel_xy[1],
                        rel_v[0],
                        rel_v[1],
                        math.sin(dhead),
                        math.cos(dhead),
                        1.0,
                        base_feat[0],
                        base_feat[1],
                        base_feat[2],
                        base_feat[3],
                        base_feat[4],
                        base_feat[5],
                    ],
                    dtype=np.float32,
                )
                hist_valid[h_idx] = 1.0

            target = np.zeros((F, 4), dtype=np.float32)
            target_valid = np.zeros((F,), dtype=np.float32)
            for f_idx, s in enumerate(range(anchor_t + 1, anchor_t + F + 1)):
                if s >= n_steps or not valid[i, s]:
                    continue

                rel_xy = to_local(np.array([x[i, s], y[i, s]], dtype=np.float32) - anchor_xy, anchor_heading)
                dhead = wrap_angle(float(heading[i, s]) - anchor_heading)
                target[f_idx] = np.array([rel_xy[0], rel_xy[1], math.sin(dhead), math.cos(dhead)], dtype=np.float32)
                target_valid[f_idx] = 1.0

            valid_future_steps = int(np.sum(target_valid))
            if anchor_t == current_time_index:
                if valid_future_steps < 1:
                    continue
            else:
                if valid_future_steps < int(min_future_valid):
                    continue

            nbr_feat = np.zeros((K, 10), dtype=np.float32)
            nbr_valid = np.zeros((K,), dtype=np.float32)

            candidates = []
            for j in range(n_obj):
                if j == i or not valid[j, anchor_t]:
                    continue
                rel_xy_j = to_local(
                    np.array([x[j, anchor_t], y[j, anchor_t]], dtype=np.float32) - anchor_xy, anchor_heading
                )
                dist = float(np.linalg.norm(rel_xy_j))
                candidates.append((dist, j, rel_xy_j))

            candidates.sort(key=lambda x_: x_[0])
            for k_idx, (_, j, rel_xy_j) in enumerate(candidates[:K]):
                rel_v_j = to_local(
                    np.array([vx[j, anchor_t] - vx[i, anchor_t], vy[j, anchor_t] - vy[i, anchor_t]], dtype=np.float32),
                    anchor_heading,
                )
                dhead_ij = wrap_angle(float(heading[j, anchor_t]) - float(heading[i, anchor_t]))
                type_oh_j = object_type_one_hot(int(object_type[j]))

                nbr_feat[k_idx] = np.array(
                    [
                        rel_xy_j[0],
                        rel_xy_j[1],
                        rel_v_j[0],
                        rel_v_j[1],
                        math.sin(dhead_ij),
                        math.cos(dhead_ij),
                        type_oh_j[0],
                        type_oh_j[1],
                        type_oh_j[2],
                        1.0,
                    ],
                    dtype=np.float32,
                )
                nbr_valid[k_idx] = 1.0

            map_feat = compute_map_context(anchor_xy, anchor_heading, map_cache)
            map_valid = np.array([map_feat[-1]], dtype=np.float32)

            static_feat = np.concatenate(
                [
                    type_oh,
                    size_vec,
                    np.array([np.linalg.norm(anchor_v)], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)

            meta = np.array(
                [
                    scenario_hash,
                    float(object_id[i]),
                    float(anchor_t),
                    float(anchor_xy[0]),
                    float(anchor_xy[1]),
                    float(anchor_heading),
                    float(anchor_z),
                ],
                dtype=np.float32,
            )
            # Emphasize difficult driving situations during training
            # If the vehicle is moving very fast or is dangerously close to a road edge,
            # duplicate this sample in the final dataset to force the model to pay attention.
            anchor_speed = float(np.linalg.norm(anchor_v))
            is_high_speed = anchor_speed >= float(high_speed_step_threshold)
            is_near_edge = (map_feat[-1] > 0.0) and (map_feat[3] <= float(edge_near_threshold_m))
            repeat_count = int(hard_case_duplication_factor) if (is_high_speed or is_near_edge) else 1

            for _ in range(repeat_count):
                hist_list.append(hist_feat)
                nbr_list.append(nbr_feat)
                map_list.append(map_feat)
                static_list.append(static_feat)
                target_list.append(target)
                hist_valid_list.append(hist_valid)
                target_valid_list.append(target_valid)
                nbr_valid_list.append(nbr_valid)
                map_valid_list.append(map_valid)
                meta_list.append(meta)

    if not hist_list:
        return _empty_sample_dict(H, F, K)

    return {
        "hist": torch.from_numpy(np.stack(hist_list)).float(),
        "nbr": torch.from_numpy(np.stack(nbr_list)).float(),
        "map": torch.from_numpy(np.stack(map_list)).float(),
        "static": torch.from_numpy(np.stack(static_list)).float(),
        "target": torch.from_numpy(np.stack(target_list)).float(),
        "masks": {
            "hist_valid": torch.from_numpy(np.stack(hist_valid_list)).float(),
            "target_valid": torch.from_numpy(np.stack(target_valid_list)).float(),
            "nbr_valid": torch.from_numpy(np.stack(nbr_valid_list)).float(),
            "map_valid": torch.from_numpy(np.stack(map_valid_list)).float(),
        },
        "meta": torch.from_numpy(np.stack(meta_list)).float(),
    }

def _process_scenario_worker(args):
    (
        bytes_example,
        H,
        F,
        K,
        current_time_index,
        challenge_type_val,
        anchor_stride,
        min_future_valid,
        multi_anchor,
        high_speed_step_threshold,
        edge_near_threshold_m,
        hard_case_duplication_factor,
    ) = args
    scenario_proto = scenario_pb2.Scenario.FromString(bytes_example)
    return build_training_samples_from_scenario(
        scenario_proto,
        H=H,
        F=F,
        K=K,
        current_time_index=current_time_index,
        challenge_type_val=challenge_type_val,
        anchor_stride=anchor_stride,
        min_future_valid=min_future_valid,
        multi_anchor=multi_anchor,
        high_speed_step_threshold=high_speed_step_threshold,
        edge_near_threshold_m=edge_near_threshold_m,
        hard_case_duplication_factor=hard_case_duplication_factor,
    )

def _append_to_buffer(buffer: dict, sample_dict: dict) -> int:
    n = int(sample_dict["hist"].shape[0])
    if n == 0:
        return 0

    for key in ["hist", "nbr", "map", "static", "target", "meta"]:
        buffer[key].append(sample_dict[key])
    for key in ["hist_valid", "target_valid", "nbr_valid", "map_valid"]:
        buffer["masks"][key].append(sample_dict["masks"][key])
    return n

def _flush_buffer(buffer: dict, out_dir: Path, shard_idx: int) -> tuple[str, int]:
    shard = {
        key: torch.cat(buffer[key], dim=0)
        for key in ["hist", "nbr", "map", "static", "target", "meta"]
    }
    shard["masks"] = {
        key: torch.cat(buffer["masks"][key], dim=0)
        for key in ["hist_valid", "target_valid", "nbr_valid", "map_valid"]
    }

    shard_path = out_dir / f"samples_{shard_idx:05d}.pt"
    torch.save(shard, shard_path)

    n_written = int(shard["hist"].shape[0])

    for key in ["hist", "nbr", "map", "static", "target", "meta"]:
        buffer[key].clear()
    for key in ["hist_valid", "target_valid", "nbr_valid", "map_valid"]:
        buffer["masks"][key].clear()

    return str(shard_path), n_written

def write_sample_shards(
    streamer,
    out_dir: Path,
    challenge_type_val: int,
    max_scenarios: int,
    shard_size: int,
    max_cache_gb: float,
    H: int,
    F: int,
    K: int,
    num_workers: int,
    anchor_stride: int,
    min_future_valid: int,
    multi_anchor: bool,
    high_speed_step_threshold: float,
    edge_near_threshold_m: float,
    hard_case_duplication_factor: int,
) -> list[str]:
    """
    Orchestrates the parallel processing of scenarios and writes them to disk.
    Uses multiprocessing to extract features concurrently and flushes tensor buffers to PyTorch files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    max_cache_bytes = int(max_cache_gb * (1024 ** 3))

    buffer = {
        "hist": [],
        "nbr": [],
        "map": [],
        "static": [],
        "target": [],
        "meta": [],
        "masks": {
            "hist_valid": [],
            "target_valid": [],
            "nbr_valid": [],
            "map_valid": [],
        },
    }

    shard_paths = []
    shard_idx = 0
    scenarios_seen = 0
    buffered_samples = 0
    total_samples = 0
    stopped_for_budget = False
    anchor_counts: dict[int, int] = {}

    existing_bytes = directory_size_bytes(out_dir)
    if existing_bytes >= max_cache_bytes:
        print(f"{out_dir} already uses {bytes_to_gb(existing_bytes):.2f} GB, at or above budget {max_cache_gb:.2f} GB.")
        return sorted(str(path) for path in out_dir.glob("samples_*.pt"))

    batch_size = num_workers * 2
    scenario_batch = []
    
    print(f"Processing scenarios with {num_workers} parallel workers (batch size: {batch_size})")
    first_sample_validated = False

    with mp.Pool(num_workers) as pool:
        for bytes_example in streamer:
            scenario_batch.append(bytes_example)

            if len(scenario_batch) >= batch_size:
                args_list = [
                    (
                        b,
                        H,
                        F,
                        K,
                        CURRENT_TIME_INDEX,
                        challenge_type_val,
                        anchor_stride,
                        min_future_valid,
                        multi_anchor,
                        high_speed_step_threshold,
                        edge_near_threshold_m,
                        hard_case_duplication_factor,
                    )
                    for b in scenario_batch
                ]
                
                results = pool.map(_process_scenario_worker, args_list)

                for sample_dict in results:
                    added = _append_to_buffer(buffer, sample_dict)
                    buffered_samples += added
                    total_samples += added
                    if added > 0:
                        anchor_vals = sample_dict["meta"][:, 2].to(torch.int64).tolist()
                        for anchor_t in anchor_vals:
                            anchor_t = int(anchor_t)
                            anchor_counts[anchor_t] = anchor_counts.get(anchor_t, 0) + 1
                    
                    if not first_sample_validated and added > 0:
                        print(f"\n=== First Sample Validation ===")
                        print(f"hist shape: {sample_dict['hist'].shape}")
                        print(f"target shape: {sample_dict['target'].shape}")
                        print(f"=== Validation Complete ===\n")
                        first_sample_validated = True

                scenarios_seen += len(scenario_batch)
                scenario_batch.clear()

                if buffered_samples >= shard_size and buffer["hist"]:
                    shard_path, n_written = _flush_buffer(buffer, out_dir, shard_idx)
                    shard_paths.append(shard_path)
                    shard_idx += 1
                    buffered_samples = 0

                    current_bytes = directory_size_bytes(out_dir)
                    print(
                        f"Wrote shard {shard_idx} with {n_written} samples "
                        f"(scenarios: {scenarios_seen}, cache size: {bytes_to_gb(current_bytes):.2f} GB / {max_cache_gb:.2f} GB)"
                    )
                    if current_bytes >= max_cache_bytes:
                        stopped_for_budget = True
                        print(f"Stopping {out_dir.name} split because cache budget was reached.")
                        break

            if scenarios_seen >= max_scenarios:
                break

        if scenario_batch and not stopped_for_budget:
            args_list = [
                (
                    b,
                    H,
                    F,
                    K,
                    CURRENT_TIME_INDEX,
                    challenge_type_val,
                    anchor_stride,
                    min_future_valid,
                    multi_anchor,
                    high_speed_step_threshold,
                    edge_near_threshold_m,
                    hard_case_duplication_factor,
                )
                for b in scenario_batch
            ]
            results = pool.map(_process_scenario_worker, args_list)

            for sample_dict in results:
                added = _append_to_buffer(buffer, sample_dict)
                buffered_samples += added
                total_samples += added
                if added > 0:
                    anchor_vals = sample_dict["meta"][:, 2].to(torch.int64).tolist()
                    for anchor_t in anchor_vals:
                        anchor_t = int(anchor_t)
                        anchor_counts[anchor_t] = anchor_counts.get(anchor_t, 0) + 1

            scenarios_seen += len(scenario_batch)
            scenario_batch.clear()

    if buffer["hist"] and not stopped_for_budget:
        shard_path, n_written = _flush_buffer(buffer, out_dir, shard_idx)
        shard_paths.append(shard_path)
        current_bytes = directory_size_bytes(out_dir)
        print(
            f"Wrote final shard with {n_written} samples "
            f"(cache size: {bytes_to_gb(current_bytes):.2f} GB / {max_cache_gb:.2f} GB)"
        )

    manifest = {
        "max_scenarios": max_scenarios,
        "scenarios_seen": scenarios_seen,
        "total_samples": total_samples,
        "H": H,
        "F": F,
        "K": K,
        "anchor_stride": anchor_stride,
        "min_future_valid": min_future_valid,
        "multi_anchor": bool(multi_anchor),
        "high_speed_step_threshold": float(high_speed_step_threshold),
        "edge_near_threshold_m": float(edge_near_threshold_m),
        "hard_case_duplication_factor": int(hard_case_duplication_factor),
        "anchor_counts": anchor_counts,
        "shards": sorted(str(path) for path in out_dir.glob("samples_*.pt")),
        "cache_size_gb": bytes_to_gb(directory_size_bytes(out_dir)),
        "cache_budget_gb": max_cache_gb,
        "stopped_for_budget": stopped_for_budget,
    }
    torch.save(manifest, out_dir / "manifest.pt")

    print(
        f"Shard build complete for {out_dir.name}. "
        f"scenarios={scenarios_seen}, samples={total_samples}, shards={len(manifest['shards'])}, "
        f"size={manifest['cache_size_gb']:.2f} GB"
    )
    return manifest["shards"]

def validate_cache_split(cache_dir: Path, expected_h: int, expected_f: int, expected_k: int) -> None:
    shard_paths = sorted(cache_dir.glob("samples_*.pt"))
    assert shard_paths, f"No shards found in {cache_dir}"

    shard = torch.load(shard_paths[0], map_location="cpu", weights_only=False)

    assert int(shard["hist"].shape[1]) == expected_h
    assert int(shard["target"].shape[1]) == expected_f
    assert int(shard["nbr"].shape[1]) == expected_k
    for key in ["hist", "nbr", "map", "static", "target", "meta"]:
        assert torch.isfinite(shard[key]).all(), f"Non-finite values found in {cache_dir}/{key}"
    for key in ["hist_valid", "target_valid", "nbr_valid", "map_valid"]:
        assert torch.isfinite(shard["masks"][key]).all(), f"Non-finite mask values found in {cache_dir}/{key}"

    manifest_path = cache_dir / "manifest.pt"
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    anchor_counts = manifest.get("anchor_counts", {})
    if bool(manifest.get("multi_anchor", False)):
        expected_anchors = list(range(ANCHOR_MIN_T, ANCHOR_MAX_T + 1, manifest.get("anchor_stride", ANCHOR_STRIDE)))
        missing_anchors = [a for a in expected_anchors if int(anchor_counts.get(a, 0)) <= 0]
        assert not missing_anchors, f"Missing anchor_t samples in {cache_dir}: {missing_anchors}"

    print(f"Validated {cache_dir} successfully.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    seed_everything(SEED)
    
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("TensorFlow:", tf.__version__)
    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)
    print(f"Storage budget: total={LOCAL_STORAGE_BUDGET_GB:.1f} GB, reserved free={RESERVED_FREE_SPACE_GB:.1f} GB")

    if REBUILD_EXISTING_CACHE:
        for cache_dir in [TRAIN_CACHE_DIR, VAL_CACHE_DIR]:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        print("Existing cache directories removed and recreated.")
    
    challenge_type = submission_specs.ChallengeType.SCENARIO_GEN
    submission_config = submission_specs.get_submission_config(challenge_type)
    
    train_streamer = StreamingWaymoDataset(
        GCS_TRAIN_PATH,
        MAX_TRAIN_SCENARIOS,
        FILES_PER_BATCH,
        BATCH_DIR,
        "TRAIN",
        test_mode=TEST_MODE,
    )
    val_streamer = StreamingWaymoDataset(
        GCS_VAL_PATH,
        MAX_VAL_SCENARIOS,
        FILES_PER_BATCH,
        BATCH_DIR,
        "VAL",
        test_mode=TEST_MODE,
    )

    TRAIN_SHARDS = write_sample_shards(
        train_streamer,
        TRAIN_CACHE_DIR,
        challenge_type,
        max_scenarios=MAX_TRAIN_SCENARIOS,
        shard_size=SHARD_SIZE,
        max_cache_gb=TRAIN_CACHE_BUDGET_GB,
        H=HISTORY_STEPS,
        F=FUTURE_STEPS,
        K=NEIGHBORS_K,
        num_workers=NUM_WORKERS,
        anchor_stride=ANCHOR_STRIDE,
        min_future_valid=MIN_FUTURE_VALID_AUX,
        multi_anchor=MULTI_ANCHOR,
        high_speed_step_threshold=HIGH_SPEED_STEP_THRESHOLD,
        edge_near_threshold_m=EDGE_NEAR_THRESHOLD_M,
        hard_case_duplication_factor=HARD_CASE_DUPLICATION_FACTOR,
    )

    VAL_SHARDS = write_sample_shards(
        val_streamer,
        VAL_CACHE_DIR,
        challenge_type,
        max_scenarios=MAX_VAL_SCENARIOS,
        shard_size=max(1, SHARD_SIZE // 2),
        max_cache_gb=VAL_CACHE_BUDGET_GB,
        H=HISTORY_STEPS,
        F=FUTURE_STEPS,
        K=NEIGHBORS_K,
        num_workers=NUM_WORKERS,
        anchor_stride=ANCHOR_STRIDE,
        min_future_valid=MIN_FUTURE_VALID_AUX,
        multi_anchor=MULTI_ANCHOR,
        high_speed_step_threshold=HIGH_SPEED_STEP_THRESHOLD,
        edge_near_threshold_m=EDGE_NEAR_THRESHOLD_M,
        hard_case_duplication_factor=HARD_CASE_DUPLICATION_FACTOR,
    )

    validate_cache_split(TRAIN_CACHE_DIR, HISTORY_STEPS, FUTURE_STEPS, NEIGHBORS_K)
    validate_cache_split(VAL_CACHE_DIR, HISTORY_STEPS, FUTURE_STEPS, NEIGHBORS_K)
    print("All processing completed successfully.")
