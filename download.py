import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_TRT_LOGGER_VERBOSITY'] = '0'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import torch
from tqdm import tqdm

# Configure TensorFlow for CPU-only mode with optimized parallelism.
# Main process handles TFRecord IO so it benefits from a few more threads;
# cap at half the cores so spawned workers still have room.
tf.config.set_visible_devices([], 'GPU')
_MAIN_TF_THREADS = max(1, (os.cpu_count() or 2) // 4)
tf.config.threading.set_inter_op_parallelism_threads(_MAIN_TF_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(_MAIN_TF_THREADS)

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
NEIGHBORS_K = 16
ANCHOR_MIN_T = CURRENT_TIME_INDEX
ANCHOR_MAX_T = 70
ANCHOR_STRIDE = 5
MIN_FUTURE_VALID_AUX = 20
MULTI_ANCHOR = True

MAP_SCHEMA_VERSION = "v2"
MAP_POINT_STRIDE = 4
MAP_POLYLINE_POINTS = 64
MAP_POLYLINE_OVERLAP = 32
MAP_POLYLINE_BUDGET = 256
MAP_POINT_DIM = 11
MAP_SIGNAL_DIM = 10

# Long-tail emphasis inspired by high-speed failure analysis in challenge reports.
HIGH_SPEED_STEP_THRESHOLD = 2.6  # ~26 m/s (approx 58 mph) in 0.1s delta units
EDGE_NEAR_THRESHOLD_M = 6.0
HARD_CASE_DUPLICATION_FACTOR = 2

MAX_TRAIN_SCENARIOS = 150000  # Unlimited (1000 files × 150 scenarios each)
MAX_VAL_SCENARIOS = 150  # Only validation scenarios
FILES_PER_BATCH = 1
# v2 samples are substantially larger due to map polyline tensors.
# Keep shards smaller to avoid multi-GB in-memory buffering that can look frozen.
SHARD_SIZE = 1024
REBUILD_EXISTING_CACHE = True
TEST_MODE = False

# Auto-scale workers: leave 2 cores for the main process (GCS download + orchestration).
# Each worker uses 1 TF intra/inter thread (lightweight protobuf + eager ops), so
# total CPU usage ≈ (CPU_COUNT - 2) + 2 = CPU_COUNT with no over-subscription.
CPU_COUNT = os.cpu_count() or 1
# Conservative default for stability; raise manually after confirming throughput.
NUM_WORKERS = min(4, max(1, CPU_COUNT - 2))
PROGRESS_PRINT_EVERY = 10

# Optimized for 113 GB available storage
LOCAL_STORAGE_BUDGET_GB = 110.0  # Use 110 GB of 113 GB available
RESERVED_FREE_SPACE_GB = 3.0     # Keep 3 GB free buffer
TEMP_DOWNLOAD_BUDGET_GB = 2.0
TRAIN_CACHE_BUDGET_GB = 95.0     # Maximize training data
VAL_CACHE_BUDGET_GB = 5.0        # Minimal validation space needed

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
assert TRAIN_CACHE_BUDGET_GB + VAL_CACHE_BUDGET_GB + TEMP_DOWNLOAD_BUDGET_GB <= (LOCAL_STORAGE_BUDGET_GB - RESERVED_FREE_SPACE_GB), \
    f"Storage budget {TRAIN_CACHE_BUDGET_GB + VAL_CACHE_BUDGET_GB + TEMP_DOWNLOAD_BUDGET_GB} GB exceeds available {LOCAL_STORAGE_BUDGET_GB - RESERVED_FREE_SPACE_GB} GB"

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

            first_name = os.path.basename(batch[0]) if len(batch) > 0 else "unknown"
            print(
                f"[{self.split_name}] Downloading batch ({len(batch)} file(s)) "
                f"cursor={file_cursor}/{len(self.all_files)} first={first_name}"
            )
            t0 = time.time()
            try:
                subprocess.run(
                    ["gsutil", "-m", "cp"] + batch + [str(self.batch_dir) + "/"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                stderr = (e.stderr or "").strip()
                stdout = (e.stdout or "").strip()
                msg = stderr if stderr else stdout
                raise RuntimeError(f"[{self.split_name}] gsutil copy failed: {msg[:500]}") from e
            print(
                f"[{self.split_name}] Download complete in {time.time() - t0:.1f}s "
                f"(scenarios_yielded={scenarios_yielded})"
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


MAP_TYPE_TO_INDEX = {
    "lane": 0,
    "road_line": 1,
    "road_edge": 2,
    "crosswalk": 3,
    "speed_bump": 4,
    "stop_sign": 5,
    "other": 6,
}
MAP_TYPE_DIM = 7

SIGNAL_NO_SIGNAL = 0
SIGNAL_UNKNOWN = 1
SIGNAL_CLASS_FROM_STATE = {
    0: SIGNAL_UNKNOWN,   # UNKNOWN
    1: 2,                # ARROW_STOP
    2: 3,                # ARROW_CAUTION
    3: 4,                # ARROW_GO
    4: 5,                # STOP
    5: 6,                # CAUTION
    6: 7,                # GO
    7: 8,                # FLASHING_STOP
    8: 9,                # FLASHING_CAUTION
}

def scenario_id_to_int(scenario_id: str) -> int:
    try:
        return int(scenario_id[:16], 16)
    except Exception:
        return abs(hash(scenario_id)) % (2**31 - 1)

def _extract_polyline_points(polyline) -> np.ndarray:
    if polyline is None or len(polyline) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(p.x, p.y) for p in polyline], dtype=np.float32)

def _extract_polygon_points(polygon) -> np.ndarray:
    if polygon is None or len(polygon) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.array([(p.x, p.y) for p in polygon], dtype=np.float32)
    # close polygon loop for consistent direction features.
    if pts.shape[0] > 2:
        pts = np.concatenate([pts, pts[:1]], axis=0)
    return pts


def _downsample_points(pts: np.ndarray, point_stride: int) -> np.ndarray:
    if pts.shape[0] == 0 or point_stride <= 1:
        return pts
    sampled = pts[::point_stride]
    if sampled.shape[0] == 1 and pts.shape[0] > 1:
        sampled = np.stack([pts[0], pts[-1]], axis=0)
    return sampled


def _compute_unit_dirs(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] >= 2:
        seg = pts[1:] - pts[:-1]
        seg_norm = np.linalg.norm(seg, axis=1, keepdims=True) + 1e-6
        unit = seg / seg_norm
        return np.concatenate([unit, unit[-1:]], axis=0)
    return np.array([[1.0, 0.0]], dtype=np.float32)


def _split_polyline_segments(
    pts: np.ndarray,
    seg_len: int = MAP_POLYLINE_POINTS,
    overlap: int = MAP_POLYLINE_OVERLAP,
) -> list[np.ndarray]:
    if pts.shape[0] == 0:
        return []
    if pts.shape[0] <= seg_len:
        return [pts]

    step = max(1, seg_len - overlap)
    out = []
    starts = list(range(0, pts.shape[0] - seg_len + 1, step))
    if starts[-1] != (pts.shape[0] - seg_len):
        starts.append(pts.shape[0] - seg_len)
    for start in starts:
        out.append(pts[start : start + seg_len])
    return out


def _extract_lane_states_by_time(
    scenario_proto: scenario_pb2.Scenario,
    lane_id_set: set[int],
) -> tuple[list[dict[int, int]], dict[str, int]]:
    lane_states_by_time: list[dict[int, int]] = []
    total = 0
    matched = 0

    for dyn_state in scenario_proto.dynamic_map_states:
        lane_map: dict[int, int] = {}
        for lane_state in dyn_state.lane_states:
            lane_id = int(getattr(lane_state, "lane", -1))
            state = int(getattr(lane_state, "state", 0))
            lane_map[lane_id] = state
            total += 1
            if lane_id in lane_id_set:
                matched += 1
        lane_states_by_time.append(lane_map)

    return lane_states_by_time, {"total": total, "matched": matched}


def extract_map_arrays(scenario_proto: scenario_pb2.Scenario, point_stride: int = MAP_POINT_STRIDE) -> dict:
    """
    Parses protobuf map features into two views:
    - legacy lane/edge arrays for compatibility losses/postprocess
    - segmented polylines for v2 map encoder input
    """
    lane_points = []
    lane_dirs = []
    road_edge_points = []
    map_segments = []
    centroids = []
    lane_id_set: set[int] = set()

    for map_feature in scenario_proto.map_features:
        feature_name = map_feature.WhichOneof("feature_data")
        if feature_name is None:
            continue

        feature = getattr(map_feature, feature_name)
        polyline = getattr(feature, "polyline", None)
        polygon = getattr(feature, "polygon", None)

        pts = _extract_polyline_points(polyline)
        if pts.shape[0] == 0:
            pts = _extract_polygon_points(polygon)
        if pts.shape[0] == 0:
            continue

        pts = _downsample_points(pts, point_stride=point_stride)
        if pts.shape[0] == 0:
            continue

        if feature_name in {"lane", "road_line"}:
            lane_points.append(pts)
            lane_dirs.append(_compute_unit_dirs(pts))
        elif feature_name == "road_edge":
            road_edge_points.append(pts)

        type_idx = MAP_TYPE_TO_INDEX.get(feature_name, MAP_TYPE_TO_INDEX["other"])
        feature_lane_id: int | None = None
        if feature_name == "lane":
            feature_lane_id = int(map_feature.id)
            lane_id_set.add(feature_lane_id)

        for seg_pts in _split_polyline_segments(pts):
            map_segments.append(
                {
                    "points": seg_pts.astype(np.float32),
                    "dirs": _compute_unit_dirs(seg_pts).astype(np.float32),
                    "type_idx": int(type_idx),
                    "lane_id": feature_lane_id,
                }
            )
            centroids.append(np.mean(seg_pts, axis=0))

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

    if centroids:
        centroids = np.stack(centroids).astype(np.float32)
    else:
        centroids = np.zeros((0, 2), dtype=np.float32)

    lane_states_by_time, traffic_stats = _extract_lane_states_by_time(
        scenario_proto=scenario_proto,
        lane_id_set=lane_id_set,
    )

    return {
        "lane_points": lane_points,
        "lane_dirs": lane_dirs,
        "road_edge_points": road_edge_points,
        "map_segments": map_segments,
        "centroids": centroids,
        "lane_states_by_time": lane_states_by_time,
        "traffic_stats": traffic_stats,
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


def _to_local_points(points_xy: np.ndarray, anchor_xy: np.ndarray, anchor_heading: float) -> np.ndarray:
    if points_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rel = points_xy - anchor_xy[None, :]
    c = math.cos(anchor_heading)
    s = math.sin(anchor_heading)
    lx = c * rel[:, 0] + s * rel[:, 1]
    ly = -s * rel[:, 0] + c * rel[:, 1]
    return np.stack([lx, ly], axis=1).astype(np.float32)


def _rotate_dirs_to_local(dirs_xy: np.ndarray, anchor_heading: float) -> np.ndarray:
    if dirs_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    c = math.cos(anchor_heading)
    s = math.sin(anchor_heading)
    lx = c * dirs_xy[:, 0] + s * dirs_xy[:, 1]
    ly = -s * dirs_xy[:, 0] + c * dirs_xy[:, 1]
    return np.stack([lx, ly], axis=1).astype(np.float32)


def build_local_map_tensors(
    anchor_xy: np.ndarray,
    anchor_heading: float,
    map_cache: dict,
    anchor_t: int,
    map_budget: int = MAP_POLYLINE_BUDGET,
    polyline_points: int = MAP_POLYLINE_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    map_polyline = np.zeros((map_budget, polyline_points, MAP_POINT_DIM), dtype=np.float32)
    map_point_valid = np.zeros((map_budget, polyline_points), dtype=np.float32)
    map_polyline_valid = np.zeros((map_budget,), dtype=np.float32)
    map_signal = np.zeros((map_budget, MAP_SIGNAL_DIM), dtype=np.float32)
    map_signal[:, SIGNAL_NO_SIGNAL] = 1.0

    segments = map_cache.get("map_segments", [])
    centroids = map_cache.get("centroids", np.zeros((0, 2), dtype=np.float32))
    
    lane_states_by_time = map_cache.get("lane_states_by_time", [])
    if len(lane_states_by_time) > 0:
        lane_state_t = lane_states_by_time[min(max(anchor_t, 0), len(lane_states_by_time) - 1)]
    else:
        lane_state_t = {}

    if len(segments) == 0:
        return map_polyline, map_point_valid, map_polyline_valid, map_signal

    # Vectorized distance computation from anchor to all segment centroids
    d2 = np.sum((centroids - anchor_xy[None, :]) ** 2, axis=1)
    
    if len(segments) <= map_budget:
        chosen_indices = np.argsort(d2)
    else:
        # argpartition is O(N) instead of O(N log N) for argsort
        chosen_indices = np.argpartition(d2, map_budget - 1)[:map_budget]
        # sort just the top ones so the very closest are first
        chosen_indices = chosen_indices[np.argsort(d2[chosen_indices])]

    for out_idx, seg_idx in enumerate(chosen_indices):
        seg = segments[seg_idx]
        pts = seg["points"]
        dirs = seg["dirs"]
        n = min(polyline_points, pts.shape[0])
        if n <= 0:
            continue

        local_pts = _to_local_points(pts[:n], anchor_xy=anchor_xy, anchor_heading=anchor_heading)
        local_dirs = _rotate_dirs_to_local(dirs[:n], anchor_heading=anchor_heading)

        map_polyline[out_idx, :n, 0:2] = local_pts
        map_polyline[out_idx, :n, 2:4] = local_dirs
        type_idx = int(seg.get("type_idx", MAP_TYPE_TO_INDEX["other"]))
        if 0 <= type_idx < MAP_TYPE_DIM:
            map_polyline[out_idx, :n, 4 + type_idx] = 1.0
        map_point_valid[out_idx, :n] = 1.0
        map_polyline_valid[out_idx] = 1.0

        lane_id = seg.get("lane_id", None)
        map_signal[out_idx] = 0.0
        if lane_id is None or int(lane_id) not in lane_state_t:
            map_signal[out_idx, SIGNAL_NO_SIGNAL] = 1.0
        else:
            state_idx = SIGNAL_CLASS_FROM_STATE.get(int(lane_state_t[int(lane_id)]), SIGNAL_UNKNOWN)
            map_signal[out_idx, int(state_idx)] = 1.0

    return map_polyline, map_point_valid, map_polyline_valid, map_signal

def _empty_sample_dict(H: int, F: int, K: int) -> dict:
    return {
        "hist": np.zeros((0, H, 13), dtype=np.float32),
        "nbr": np.zeros((0, K, 10), dtype=np.float32),
        "map": np.zeros((0, 5), dtype=np.float32),
        "map_scalar": np.zeros((0, 5), dtype=np.float32),
        "map_polyline": np.zeros((0, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS, MAP_POINT_DIM), dtype=np.float32),
        "map_point_valid": np.zeros((0, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS), dtype=np.float32),
        "map_polyline_valid": np.zeros((0, MAP_POLYLINE_BUDGET), dtype=np.float32),
        "map_signal": np.zeros((0, MAP_POLYLINE_BUDGET, MAP_SIGNAL_DIM), dtype=np.float32),
        "static": np.zeros((0, 7), dtype=np.float32),
        "target": np.zeros((0, F, 4), dtype=np.float32),
        "masks": {
            "hist_valid": np.zeros((0, H), dtype=np.float32),
            "target_valid": np.zeros((0, F), dtype=np.float32),
            "nbr_valid": np.zeros((0, K), dtype=np.float32),
            "map_valid": np.zeros((0, 1), dtype=np.float32),
            "map_polyline_valid": np.zeros((0, MAP_POLYLINE_BUDGET), dtype=np.float32),
            "map_point_valid": np.zeros((0, MAP_POLYLINE_BUDGET, MAP_POLYLINE_POINTS), dtype=np.float32),
        },
        "meta": np.zeros((0, 7), dtype=np.float32),
        "traffic_stats": {"total": 0, "matched": 0},
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
    map_scalar_list = []
    map_polyline_list = []
    map_point_valid_list = []
    map_polyline_valid_list = []
    map_signal_list = []
    static_list = []
    target_list = []
    hist_valid_list = []
    target_valid_list = []
    nbr_valid_list = []
    map_valid_list = []
    map_polyline_mask_list = []
    map_point_mask_list = []
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

            map_scalar = compute_map_context(anchor_xy, anchor_heading, map_cache)
            map_valid = np.array([map_scalar[-1]], dtype=np.float32)
            map_polyline, map_point_valid, map_polyline_valid, map_signal = build_local_map_tensors(
                anchor_xy=anchor_xy,
                anchor_heading=anchor_heading,
                map_cache=map_cache,
                anchor_t=anchor_t,
                map_budget=MAP_POLYLINE_BUDGET,
                polyline_points=MAP_POLYLINE_POINTS,
            )

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
            is_near_edge = (map_scalar[-1] > 0.0) and (map_scalar[3] <= float(edge_near_threshold_m))
            repeat_count = int(hard_case_duplication_factor) if (is_high_speed or is_near_edge) else 1

            for _ in range(repeat_count):
                hist_list.append(hist_feat)
                nbr_list.append(nbr_feat)
                map_list.append(map_scalar)
                map_scalar_list.append(map_scalar)
                map_polyline_list.append(map_polyline)
                map_point_valid_list.append(map_point_valid)
                map_polyline_valid_list.append(map_polyline_valid)
                map_signal_list.append(map_signal)
                static_list.append(static_feat)
                target_list.append(target)
                hist_valid_list.append(hist_valid)
                target_valid_list.append(target_valid)
                nbr_valid_list.append(nbr_valid)
                map_valid_list.append(map_valid)
                map_polyline_mask_list.append(map_polyline_valid)
                map_point_mask_list.append(map_point_valid)
                meta_list.append(meta)

    if not hist_list:
        return _empty_sample_dict(H, F, K)

    return {
        "hist": np.stack(hist_list).astype(np.float32),
        "nbr": np.stack(nbr_list).astype(np.float32),
        "map": np.stack(map_list).astype(np.float32),
        "map_scalar": np.stack(map_scalar_list).astype(np.float32),
        "map_polyline": np.stack(map_polyline_list).astype(np.float32),
        "map_point_valid": np.stack(map_point_valid_list).astype(np.float32),
        "map_polyline_valid": np.stack(map_polyline_valid_list).astype(np.float32),
        "map_signal": np.stack(map_signal_list).astype(np.float32),
        "static": np.stack(static_list).astype(np.float32),
        "target": np.stack(target_list).astype(np.float32),
        "masks": {
            "hist_valid": np.stack(hist_valid_list).astype(np.float32),
            "target_valid": np.stack(target_valid_list).astype(np.float32),
            "nbr_valid": np.stack(nbr_valid_list).astype(np.float32),
            "map_valid": np.stack(map_valid_list).astype(np.float32),
            "map_polyline_valid": np.stack(map_polyline_mask_list).astype(np.float32),
            "map_point_valid": np.stack(map_point_mask_list).astype(np.float32),
        },
        "meta": np.stack(meta_list).astype(np.float32),
        "traffic_stats": map_cache.get("traffic_stats", {"total": 0, "matched": 0}),
    }

def _worker_tf_init():
    """Called once per spawned worker. Limits TF to 1 intra/inter thread so
    NUM_WORKERS workers collectively use NUM_WORKERS cores rather than
    NUM_WORKERS * 8 = over-subscribed thread storm."""
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


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

    for key in [
        "hist",
        "nbr",
        "map",
        "map_scalar",
        "map_polyline",
        "map_point_valid",
        "map_polyline_valid",
        "map_signal",
        "static",
        "target",
        "meta",
    ]:
        buffer[key].append(torch.from_numpy(sample_dict[key]))
    for key in ["hist_valid", "target_valid", "nbr_valid", "map_valid", "map_polyline_valid", "map_point_valid"]:
        buffer["masks"][key].append(torch.from_numpy(sample_dict["masks"][key]))
    return n

def _flush_buffer(buffer: dict, out_dir: Path, shard_idx: int) -> tuple[str, int]:
    shard = {
        key: torch.cat(buffer[key], dim=0)
        for key in [
            "hist",
            "nbr",
            "map",
            "map_scalar",
            "map_polyline",
            "map_point_valid",
            "map_polyline_valid",
            "map_signal",
            "static",
            "target",
            "meta",
        ]
    }
    shard["masks"] = {
        key: torch.cat(buffer["masks"][key], dim=0)
        for key in ["hist_valid", "target_valid", "nbr_valid", "map_valid", "map_polyline_valid", "map_point_valid"]
    }

    shard_path = out_dir / f"samples_{shard_idx:05d}.pt"
    torch.save(shard, shard_path)

    n_written = int(shard["hist"].shape[0])

    for key in [
        "hist",
        "nbr",
        "map",
        "map_scalar",
        "map_polyline",
        "map_point_valid",
        "map_polyline_valid",
        "map_signal",
        "static",
        "target",
        "meta",
    ]:
        buffer[key].clear()
    for key in ["hist_valid", "target_valid", "nbr_valid", "map_valid", "map_polyline_valid", "map_point_valid"]:
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
        "map_scalar": [],
        "map_polyline": [],
        "map_point_valid": [],
        "map_polyline_valid": [],
        "map_signal": [],
        "static": [],
        "target": [],
        "meta": [],
        "masks": {
            "hist_valid": [],
            "target_valid": [],
            "nbr_valid": [],
            "map_valid": [],
            "map_polyline_valid": [],
            "map_point_valid": [],
        },
    }

    shard_paths = []
    shard_idx = 0
    scenarios_seen = 0
    buffered_samples = 0
    total_samples = 0
    stopped_for_budget = False
    anchor_counts: dict[int, int] = {}
    traffic_state_total = 0
    traffic_state_matched = 0

    existing_bytes = directory_size_bytes(out_dir)
    if existing_bytes >= max_cache_bytes:
        print(f"{out_dir} already uses {bytes_to_gb(existing_bytes):.2f} GB, at or above budget {max_cache_gb:.2f} GB.")
        return sorted(str(path) for path in out_dir.glob("samples_*.pt"))

    def _args_gen():
        """Lazily yields worker args from the streamer so imap_unordered never
        pre-fetches more than ~2*num_workers scenarios into memory at once."""
        for bytes_example in streamer:
            yield (
                bytes_example,
                H, F, K,
                CURRENT_TIME_INDEX,
                challenge_type_val,
                anchor_stride,
                min_future_valid,
                multi_anchor,
                high_speed_step_threshold,
                edge_near_threshold_m,
                hard_case_duplication_factor,
            )

    print(f"Processing scenarios with {num_workers} parallel workers (CPU_COUNT={CPU_COUNT})")
    print(f"Max scenarios: {max_scenarios}, Storage budget: {max_cache_gb:.1f} GB")
    first_sample_validated = False
    start_time = time.time()

    # maxtasksperchild recycles workers periodically to reclaim TF memory growth.
    with mp.Pool(
        num_workers,
        initializer=_worker_tf_init,
        maxtasksperchild=500,
    ) as pool:
        
        last_print_seen = 0

        # imap_unordered keeps all workers busy: results are yielded as each
        # worker finishes rather than waiting for a whole batch to complete.
        # chunksize=1 ensures the generator is consumed lazily (no pre-fetching
        # the entire dataset into the task queue).
        for sample_dict in pool.imap_unordered(_process_scenario_worker, _args_gen(), chunksize=1):
            tstats = sample_dict.get("traffic_stats", {"total": 0, "matched": 0})
            traffic_state_total += int(tstats.get("total", 0))
            traffic_state_matched += int(tstats.get("matched", 0))
            added = _append_to_buffer(buffer, sample_dict)
            buffered_samples += added
            total_samples += added
            scenarios_seen += 1

            if scenarios_seen - last_print_seen >= PROGRESS_PRINT_EVERY or scenarios_seen == max_scenarios:
                elapsed = max(1e-6, time.time() - start_time)
                scen_per_sec = scenarios_seen / elapsed
                print(
                    f"Processed {scenarios_seen}/{max_scenarios} scenarios "
                    f"(samples={total_samples}, rate={scen_per_sec:.2f} scen/s)"
                )
                last_print_seen = scenarios_seen

            if added > 0:
                anchor_src = sample_dict["meta"][:, 2]
                if isinstance(anchor_src, torch.Tensor):
                    anchor_vals = anchor_src.to(torch.int64).tolist()
                else:
                    anchor_vals = np.asarray(anchor_src, dtype=np.int64).tolist()
                for anchor_t in anchor_vals:
                    anchor_counts[int(anchor_t)] = anchor_counts.get(int(anchor_t), 0) + 1

            if not first_sample_validated and added > 0:
                print(f"\n=== First Sample Validation ===")
                print(f"hist shape: {sample_dict['hist'].shape}")
                print(f"map_polyline shape: {sample_dict['map_polyline'].shape}")
                print(f"target shape: {sample_dict['target'].shape}")
                print(f"=== Validation Complete ===\n")
                first_sample_validated = True

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

    if buffer["hist"] and not stopped_for_budget:
        shard_path, n_written = _flush_buffer(buffer, out_dir, shard_idx)
        shard_paths.append(shard_path)
        current_bytes = directory_size_bytes(out_dir)
        print(
            f"Wrote final shard with {n_written} samples "
            f"(cache size: {bytes_to_gb(current_bytes):.2f} GB / {max_cache_gb:.2f} GB)"
        )

    manifest = {
        "schema_version": MAP_SCHEMA_VERSION,
        "max_scenarios": max_scenarios,
        "scenarios_seen": scenarios_seen,
        "total_samples": total_samples,
        "H": H,
        "F": F,
        "K": K,
        "neighbors_k": K,
        "point_stride": MAP_POINT_STRIDE,
        "polyline_points": MAP_POLYLINE_POINTS,
        "polyline_overlap": MAP_POLYLINE_OVERLAP,
        "polyline_budget": MAP_POLYLINE_BUDGET,
        "anchor_stride": anchor_stride,
        "min_future_valid": min_future_valid,
        "multi_anchor": bool(multi_anchor),
        "high_speed_step_threshold": float(high_speed_step_threshold),
        "edge_near_threshold_m": float(edge_near_threshold_m),
        "hard_case_duplication_factor": int(hard_case_duplication_factor),
        "traffic_light_state_total": int(traffic_state_total),
        "traffic_light_state_matched": int(traffic_state_matched),
        "traffic_lane_match_rate": float(traffic_state_matched / max(1, traffic_state_total)),
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
    assert int(shard["map_polyline"].shape[1]) == MAP_POLYLINE_BUDGET
    assert int(shard["map_polyline"].shape[2]) == MAP_POLYLINE_POINTS
    assert int(shard["map_polyline"].shape[3]) == MAP_POINT_DIM
    assert int(shard["map_signal"].shape[2]) == MAP_SIGNAL_DIM
    for key in [
        "hist",
        "nbr",
        "map",
        "map_scalar",
        "map_polyline",
        "map_point_valid",
        "map_polyline_valid",
        "map_signal",
        "static",
        "target",
        "meta",
    ]:
        assert torch.isfinite(shard[key]).all(), f"Non-finite values found in {cache_dir}/{key}"
    for key in ["hist_valid", "target_valid", "nbr_valid", "map_valid", "map_polyline_valid", "map_point_valid"]:
        assert torch.isfinite(shard["masks"][key]).all(), f"Non-finite mask values found in {cache_dir}/{key}"

    manifest_path = cache_dir / "manifest.pt"
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    assert str(manifest.get("schema_version", "")) == MAP_SCHEMA_VERSION, (
        f"Expected schema_version={MAP_SCHEMA_VERSION} in {manifest_path}, got {manifest.get('schema_version')}"
    )
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
                t0 = time.time()
                print(f"Removing existing cache dir: {cache_dir}")
                shutil.rmtree(cache_dir)
                print(f"Removed {cache_dir} in {time.time() - t0:.1f}s")
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
