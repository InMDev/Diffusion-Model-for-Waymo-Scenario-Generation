"""Waymo feature and rollout utilities shared by inference/EDA notebooks.

This module consolidates map/trajectory feature extraction and post-processing
logic so notebooks stay presentation-focused.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from typing import Literal

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs

from notebooks_lib.diffusion_core import sample_future_chunk

TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 3

HISTORY_STEPS = 11
FUTURE_STEPS = 80
NEIGHBORS_K = 6

DIFFUSION_SAMPLE_STEPS = 50
GUIDANCE_SWEEP = [1.0, 1.2, 1.5]
HIGHWAY_GUIDANCE_SWEEP = [1.2, 1.5, 1.8]
HIGHWAY_SAMPLE_STEPS = 64
HIGHWAY_CLAMP_SCALE = 0.8
HIGH_SPEED_STEP_THRESHOLD = 2.6

GCS_BASE = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"
GCS_VAL_PATH = f"{GCS_BASE}/validation/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIFFUSION_CFG: dict = {}

def wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))

def to_local(vec_xy: np.ndarray, anchor_heading: float) -> np.ndarray:
    c = math.cos(anchor_heading)
    s = math.sin(anchor_heading)
    x, y = vec_xy
    return np.array([c * x + s * y, -s * x + c * y], dtype=np.float32)

def to_world(local_xy: np.ndarray, anchor_heading: float) -> np.ndarray:
    c = math.cos(anchor_heading)
    s = math.sin(anchor_heading)
    lx, ly = local_xy
    return np.array([c * lx - s * ly, s * lx + c * ly], dtype=np.float32)

def object_type_one_hot(object_type: int) -> np.ndarray:
    if object_type == TYPE_VEHICLE:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if object_type == TYPE_PEDESTRIAN:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if object_type == TYPE_CYCLIST:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def _extract_polyline_points(polyline) -> np.ndarray:
    if polyline is None or len(polyline) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(p.x, p.y) for p in polyline], dtype=np.float32)

def extract_map_arrays(scenario_proto: scenario_pb2.Scenario, point_stride: int = 5) -> dict:
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
        lane_dist2 = np.sum(lane_delta**2, axis=1)
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
        edge_dist2 = np.sum(edge_delta**2, axis=1)
        dist_edge = float(np.sqrt(np.min(edge_dist2)))
        map_valid = 1.0

    return np.array([dist_lane, lane_sin, lane_cos, dist_edge, map_valid], dtype=np.float32)

def get_sim_agent_trajectories(scenario_proto: scenario_pb2.Scenario, challenge_type):
    full = trajectory_utils.ObjectTrajectories.from_scenario(scenario_proto)
    sim_ids = tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario_proto, challenge_type))
    return full.gather_objects_by_id(sim_ids)

def build_scene_rollout_features(scenario_proto: scenario_pb2.Scenario, challenge_type, submission_config) -> dict:
    trajectories = get_sim_agent_trajectories(scenario_proto, challenge_type)

    x = trajectories.x.numpy().astype(np.float32)
    y = trajectories.y.numpy().astype(np.float32)
    z = trajectories.z.numpy().astype(np.float32)
    h = trajectories.heading.numpy().astype(np.float32)
    valid = trajectories.valid.numpy().astype(bool)

    length = trajectories.length.numpy().astype(np.float32)
    width = trajectories.width.numpy().astype(np.float32)
    height = trajectories.height.numpy().astype(np.float32)

    states = np.stack([x, y, z, h], axis=-1)
    sizes = np.stack([length, width, height], axis=-1)

    return {
        "states": states,
        "sizes": sizes,
        "valid": valid,
        "object_type": trajectories.object_type.numpy().astype(np.int32),
        "object_id": trajectories.object_id,
        "map_cache": extract_map_arrays(scenario_proto),
        "num_rollouts": int(submission_config.n_rollouts),
    }

TYPE_CLAMPS = {
    TYPE_VEHICLE: {"max_step_xy": 3.0, "max_yaw": 0.10},
    TYPE_PEDESTRIAN: {"max_step_xy": 1.2, "max_yaw": 0.45},
    TYPE_CYCLIST: {"max_step_xy": 2.0, "max_yaw": 0.25},
}
DEFAULT_CLAMP = {"max_step_xy": 2.0, "max_yaw": 0.25}

# Vehicle-physics post-process constants
DT_DEFAULT = 0.1
VEH_ACCEL_MIN = -3.5
VEH_ACCEL_MAX = 2.5
VEH_STEER_MIN = -0.5
VEH_STEER_MAX = 0.5
NO_SLIP_LAT_RATIO = 0.04
MIN_EDGE_CLEARANCE_M = 1.2
LANE_PULL_GAIN = 0.20
REPULSION_MAX_PUSH = 0.35

try:
    import pandas as pd
except Exception:
    pd = None

DEFAULT_INFERENCE_CFG = {
    "guidance_values": (1.0, 1.2),
    "sample_steps": 40,
    "use_ema": True,
    "postprocess_mode": "bicycle",
    "use_history_speed_clamp": True,
    "use_road_bound": True,
    "use_light_repulsion": True,
    "dt": DT_DEFAULT,
    "use_clamping": True,
    "use_guidance_sweep": True,
    "sanitize_prediction": True,
    "force_highway_mode": False,
    "clamp_scale_override": None,
    "fixed_guidance": None,
}

DEFAULT_PHYSICS_CFG = {
    "veh_accel_min": -3.0,
    "veh_accel_max": 2.2,
    "veh_steer_limit": 0.45,
    "no_slip_lat_ratio": 0.02,
    "min_edge_clearance_m": 0.9,
    "lane_pull_gain": 0.08,
    "repulsion_max_push": 0.18,
}

VAL_FILE_CACHE: list[str] | None = None
LOCAL_WAYMO_CACHE_DIR = "./waymo_val_cache"
os.makedirs(LOCAL_WAYMO_CACHE_DIR, exist_ok=True)


def local_cache_path_for_gcs_file(gcs_path: str, cache_dir: str = LOCAL_WAYMO_CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, os.path.basename(gcs_path))


def ensure_local_tfrecord(
    gcs_path: str,
    cache_dir: str = LOCAL_WAYMO_CACHE_DIR,
    force_refresh: bool = False,
    verbose: bool = True,
) -> str:
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


def resolve_inference_cfg(inference_cfg: dict | None = None) -> dict:
    cfg = dict(DEFAULT_INFERENCE_CFG)
    if inference_cfg:
        cfg.update(inference_cfg)

    cfg["guidance_values"] = tuple(float(v) for v in cfg.get("guidance_values", tuple(GUIDANCE_SWEEP)))
    cfg["sample_steps"] = int(cfg.get("sample_steps", DIFFUSION_SAMPLE_STEPS))
    cfg["use_ema"] = bool(cfg.get("use_ema", True))
    cfg["postprocess_mode"] = str(cfg.get("postprocess_mode", "bicycle"))
    cfg["use_history_speed_clamp"] = bool(cfg.get("use_history_speed_clamp", True))
    cfg["use_road_bound"] = bool(cfg.get("use_road_bound", True))
    cfg["use_light_repulsion"] = bool(cfg.get("use_light_repulsion", True))
    cfg["dt"] = float(cfg.get("dt", DT_DEFAULT))
    cfg["use_clamping"] = bool(cfg.get("use_clamping", True))
    cfg["use_guidance_sweep"] = bool(cfg.get("use_guidance_sweep", True))
    cfg["sanitize_prediction"] = bool(cfg.get("sanitize_prediction", True))
    return cfg


def resolve_physics_cfg(physics_cfg: dict | None = None) -> dict:
    cfg = dict(DEFAULT_PHYSICS_CFG)
    if physics_cfg:
        cfg.update(physics_cfg)

    accel_min = float(cfg.get("veh_accel_min", VEH_ACCEL_MIN))
    accel_max = float(cfg.get("veh_accel_max", VEH_ACCEL_MAX))
    if accel_min > accel_max:
        accel_min, accel_max = accel_max, accel_min

    steer_limit = float(cfg.get("veh_steer_limit", max(abs(VEH_STEER_MIN), abs(VEH_STEER_MAX))))
    steer_limit = max(0.1, steer_limit)

    cfg["veh_accel_min"] = accel_min
    cfg["veh_accel_max"] = accel_max
    cfg["veh_steer_limit"] = steer_limit
    cfg["no_slip_lat_ratio"] = float(max(0.0, cfg.get("no_slip_lat_ratio", NO_SLIP_LAT_RATIO)))
    cfg["min_edge_clearance_m"] = float(max(0.0, cfg.get("min_edge_clearance_m", MIN_EDGE_CLEARANCE_M)))
    cfg["lane_pull_gain"] = float(max(0.0, cfg.get("lane_pull_gain", LANE_PULL_GAIN)))
    cfg["repulsion_max_push"] = float(max(0.0, cfg.get("repulsion_max_push", REPULSION_MAX_PUSH)))
    return cfg


def sanitize_scene_arrays(states: np.ndarray, sizes: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    sizes = np.nan_to_num(sizes, nan=0.0, posinf=0.0, neginf=0.0)

    n_obj, n_steps, _ = states.shape
    for i in range(n_obj):
        valid_idx = np.flatnonzero(valid[i])
        if valid_idx.size == 0:
            continue

        first_valid = int(valid_idx[0])
        states[i, :first_valid] = states[i, first_valid]
        sizes[i, :first_valid] = sizes[i, first_valid]

        last_seen = first_valid
        for t in range(first_valid + 1, n_steps):
            if valid[i, t]:
                last_seen = t
            else:
                states[i, t] = states[i, last_seen]
                sizes[i, t] = sizes[i, last_seen]

    return states, sizes


def apply_type_aware_clamps(
    prev_state: np.ndarray,
    proposed_state: np.ndarray,
    object_type: int,
    clamp_scale: float = 1.0,
    speed_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    clamp = TYPE_CLAMPS.get(int(object_type), DEFAULT_CLAMP)
    max_step_xy = float(clamp["max_step_xy"]) * float(clamp_scale)
    max_yaw = float(clamp["max_yaw"]) * float(clamp_scale)

    delta_xy = proposed_state[:2] - prev_state[:2]
    speed = float(np.linalg.norm(delta_xy))

    # Optional history-based speed band from observed context.
    if speed_bounds is not None:
        speed_min, speed_max = speed_bounds
        speed_min = max(0.0, float(speed_min))
        speed_max = max(speed_min + 1e-3, float(speed_max))
        target_speed = float(np.clip(speed, speed_min, speed_max))
        if speed > 1e-6:
            delta_xy = delta_xy / speed * target_speed
            speed = target_speed

    if speed > max_step_xy and speed > 1e-6:
        delta_xy = delta_xy / speed * max_step_xy

    new_xy = prev_state[:2] + delta_xy

    dhead = wrap_angle(float(proposed_state[3]) - float(prev_state[3]))
    dhead = float(np.clip(dhead, -max_yaw, max_yaw))
    new_heading = wrap_angle(float(prev_state[3]) + dhead)

    return np.array([new_xy[0], new_xy[1], prev_state[2], new_heading], dtype=np.float32)


def build_rollout_condition_batch(
    states: np.ndarray,
    valid: np.ndarray,
    object_type: np.ndarray,
    sizes: np.ndarray,
    map_cache: dict,
    current_t: int,
    H: int = HISTORY_STEPS,
    K: int = NEIGHBORS_K,
) -> dict:
    n_obj, n_steps, _ = states.shape

    vel = np.zeros((n_obj, n_steps, 2), dtype=np.float32)
    vel[:, 1:, :] = states[:, 1:, :2] - states[:, :-1, :2]

    hist = np.zeros((n_obj, H, 13), dtype=np.float32)
    nbr = np.zeros((n_obj, K, 10), dtype=np.float32)
    map_feat = np.zeros((n_obj, 5), dtype=np.float32)
    static = np.zeros((n_obj, 7), dtype=np.float32)

    hist_valid = np.zeros((n_obj, H), dtype=np.float32)
    nbr_valid = np.zeros((n_obj, K), dtype=np.float32)
    map_valid = np.zeros((n_obj, 1), dtype=np.float32)

    safe_t = min(current_t, n_steps - 1)

    for i in range(n_obj):
        type_oh = object_type_one_hot(int(object_type[i]))
        size_vec = sizes[i, safe_t].astype(np.float32)
        static[i] = np.concatenate([type_oh, size_vec, np.array([np.linalg.norm(vel[i, safe_t])], dtype=np.float32)])

        if not valid[i, safe_t]:
            continue

        anchor_xy = states[i, safe_t, :2]
        anchor_h = float(states[i, safe_t, 3])

        for h_idx, s in enumerate(range(safe_t - H + 1, safe_t + 1)):
            base = np.concatenate([type_oh, size_vec], axis=0)
            if s < 0 or s >= n_steps or not valid[i, s]:
                hist[i, h_idx, 7:] = base
                continue

            rel_xy = to_local(states[i, s, :2] - anchor_xy, anchor_h)
            rel_v = to_local(vel[i, s], anchor_h)
            dhead = wrap_angle(float(states[i, s, 3]) - anchor_h)
            hist[i, h_idx] = np.array(
                [
                    rel_xy[0], rel_xy[1], rel_v[0], rel_v[1],
                    math.sin(dhead), math.cos(dhead), 1.0,
                    base[0], base[1], base[2], base[3], base[4], base[5],
                ],
                dtype=np.float32,
            )
            hist_valid[i, h_idx] = 1.0

        candidates = []
        for j in range(n_obj):
            if j == i or not valid[j, safe_t]:
                continue
            rel_xy_j = to_local(states[j, safe_t, :2] - anchor_xy, anchor_h)
            dist = float(np.linalg.norm(rel_xy_j))
            candidates.append((dist, j, rel_xy_j))

        candidates.sort(key=lambda x_: x_[0])
        for k_idx, (_, j, rel_xy_j) in enumerate(candidates[:K]):
            rel_v_j = to_local(vel[j, safe_t] - vel[i, safe_t], anchor_h)
            dhead = wrap_angle(float(states[j, safe_t, 3]) - float(states[i, safe_t, 3]))
            type_oh_j = object_type_one_hot(int(object_type[j]))
            nbr[i, k_idx] = np.array(
                [
                    rel_xy_j[0], rel_xy_j[1], rel_v_j[0], rel_v_j[1],
                    math.sin(dhead), math.cos(dhead),
                    type_oh_j[0], type_oh_j[1], type_oh_j[2], 1.0,
                ],
                dtype=np.float32,
            )
            nbr_valid[i, k_idx] = 1.0

        map_i = compute_map_context(anchor_xy, anchor_h, map_cache)
        map_feat[i] = map_i
        map_valid[i, 0] = map_i[-1]

    cond = {
        "hist": torch.from_numpy(hist).to(DEVICE),
        "nbr": torch.from_numpy(nbr).to(DEVICE),
        "map": torch.from_numpy(map_feat).to(DEVICE),
        "static": torch.from_numpy(static).to(DEVICE),
        "masks": {
            "hist_valid": torch.from_numpy(hist_valid).to(DEVICE),
            "target_valid": torch.ones((n_obj, FUTURE_STEPS), dtype=torch.float32, device=DEVICE),
            "nbr_valid": torch.from_numpy(nbr_valid).to(DEVICE),
            "map_valid": torch.from_numpy(map_valid).to(DEVICE),
        },
    }
    return cond


def _distance_to_nearest(points: np.ndarray, query_xy: np.ndarray) -> tuple[float, int]:
    if points.shape[0] == 0:
        return float("inf"), -1
    d2 = np.sum((points - query_xy[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return float(np.sqrt(d2[idx])), idx


def build_history_speed_bounds(
    states: np.ndarray,
    valid: np.ndarray,
    current_idx: int,
    dt: float = DT_DEFAULT,
    min_margin: float = 0.25,
    max_margin: float = 0.75,
) -> np.ndarray:
    # Returns per-agent [min_speed, max_speed] in m/s based on observed history.
    n_obj = states.shape[0]
    bounds = np.zeros((n_obj, 2), dtype=np.float32)

    for i in range(n_obj):
        hist_speeds = []
        for t in range(max(1, current_idx - 10), current_idx + 1):
            if not (valid[i, t] and valid[i, t - 1]):
                continue
            dv = states[i, t, :2] - states[i, t - 1, :2]
            hist_speeds.append(float(np.linalg.norm(dv) / max(dt, 1e-6)))

        if len(hist_speeds) == 0:
            bounds[i] = np.array([0.0, 1e6], dtype=np.float32)
            continue

        vmin = min(hist_speeds)
        vmax = max(hist_speeds)
        bounds[i] = np.array([
            max(0.0, vmin - min_margin),
            max(vmin + 0.2, vmax + max_margin),
        ], dtype=np.float32)

    return bounds


def estimate_vehicle_initial_dynamics(
    states: np.ndarray,
    valid: np.ndarray,
    current_idx: int,
    object_type: np.ndarray,
    sizes: np.ndarray,
    dt: float = DT_DEFAULT,
) -> dict:
    n_obj = states.shape[0]
    v0 = np.zeros((n_obj,), dtype=np.float32)
    a0 = np.zeros((n_obj,), dtype=np.float32)
    yaw_rate0 = np.zeros((n_obj,), dtype=np.float32)
    direction_sign = np.ones((n_obj,), dtype=np.float32)
    wheelbase = np.full((n_obj,), 2.7, dtype=np.float32)
    speed_bounds = build_history_speed_bounds(states, valid, current_idx, dt=dt)

    for i in range(n_obj):
        if int(object_type[i]) != TYPE_VEHICLE:
            continue

        length = float(sizes[i, min(current_idx, sizes.shape[1] - 1), 0])
        wheelbase[i] = float(np.clip(0.6 * length, 2.2, 3.4))

        signed_speeds = []
        yaw_rates = []
        for t in range(max(1, current_idx - 10), current_idx + 1):
            if not (valid[i, t] and valid[i, t - 1]):
                continue
            dxy = states[i, t, :2] - states[i, t - 1, :2]
            psi_prev = float(states[i, t - 1, 3])
            forward_axis = np.array([math.cos(psi_prev), math.sin(psi_prev)], dtype=np.float32)
            signed_speed = float(np.dot(dxy, forward_axis) / max(dt, 1e-6))
            signed_speeds.append(signed_speed)

            dpsi = wrap_angle(float(states[i, t, 3]) - psi_prev)
            yaw_rates.append(float(dpsi / max(dt, 1e-6)))

        if len(signed_speeds) > 0:
            v0[i] = float(np.mean(signed_speeds[-3:]))
            direction_sign[i] = 1.0 if v0[i] >= 0.0 else -1.0
            if len(signed_speeds) > 1:
                a0[i] = float((signed_speeds[-1] - signed_speeds[-2]) / max(dt, 1e-6))
        if len(yaw_rates) > 0:
            yaw_rate0[i] = float(np.mean(yaw_rates[-3:]))

    return {
        "v0": v0,
        "a0": a0,
        "yaw_rate0": yaw_rate0,
        "direction_sign": direction_sign,
        "wheelbase": wheelbase,
        "speed_bounds": speed_bounds,
    }


def local_prediction_to_controls(
    current_state: np.ndarray,
    desired_xy: np.ndarray,
    desired_heading: float,
    v_curr: float,
    wheelbase: float,
    dt: float = DT_DEFAULT,
    physics_cfg: dict | None = None,
) -> tuple[float, float]:
    cfg = resolve_physics_cfg(physics_cfg)
    accel_min = float(cfg["veh_accel_min"])
    accel_max = float(cfg["veh_accel_max"])
    steer_lim = float(cfg["veh_steer_limit"])

    x, y, _, psi = current_state
    disp = desired_xy - np.array([x, y], dtype=np.float32)

    c = math.cos(float(psi))
    s = math.sin(float(psi))
    along = float(c * disp[0] + s * disp[1])
    lat = float(-s * disp[0] + c * disp[1])

    desired_speed = along / max(dt, 1e-6)
    desired_speed = float(np.clip(desired_speed, -15.0, 40.0))
    a_cmd = float(np.clip((desired_speed - v_curr) / max(dt, 1e-6), accel_min, accel_max))

    lookahead = max(2.0, abs(v_curr) * dt * 2.0 + 1.0)
    alpha = math.atan2(lat, max(1e-3, along))
    delta_pp = math.atan2(2.0 * max(wheelbase, 1e-3) * math.sin(alpha), lookahead)
    heading_err = wrap_angle(float(desired_heading) - float(psi))
    delta_cmd = 0.7 * delta_pp + 0.3 * heading_err
    delta_cmd = float(np.clip(delta_cmd, -steer_lim, steer_lim))

    return a_cmd, delta_cmd

def enforce_road_bound(
    state_xy: np.ndarray,
    map_cache: dict,
    min_edge_clearance: float | None = None,
    lane_pull_gain: float | None = None,
    physics_cfg: dict | None = None,
) -> np.ndarray:
    cfg = resolve_physics_cfg(physics_cfg)
    if min_edge_clearance is None:
        min_edge_clearance = float(cfg["min_edge_clearance_m"])
    if lane_pull_gain is None:
        lane_pull_gain = float(cfg["lane_pull_gain"])

    xy = state_xy.astype(np.float32).copy()

    edge_pts = map_cache.get("road_edge_points", np.zeros((0, 2), dtype=np.float32))
    lane_pts = map_cache.get("lane_points", np.zeros((0, 2), dtype=np.float32))

    def _project_away_from_edge(xy_in: np.ndarray) -> np.ndarray:
        if edge_pts.shape[0] == 0:
            return xy_in
        edge_dist, edge_idx = _distance_to_nearest(edge_pts, xy_in)
        if edge_idx < 0 or edge_dist >= min_edge_clearance:
            return xy_in
        edge_xy = edge_pts[edge_idx]
        away = xy_in - edge_xy
        n = float(np.linalg.norm(away))
        if n < 1e-4:
            away = np.array([1.0, 0.0], dtype=np.float32)
            n = 1.0
        away /= n
        return edge_xy + away * float(min_edge_clearance)

    xy = _project_away_from_edge(xy)

    if lane_pts.shape[0] > 0:
        _, lane_idx = _distance_to_nearest(lane_pts, xy)
        if lane_idx >= 0:
            lane_xy = lane_pts[lane_idx]
            xy = xy + float(lane_pull_gain) * (lane_xy - xy)

    xy = _project_away_from_edge(xy)
    return xy

def _apply_no_slip_update(
    prev_xy: np.ndarray,
    psi_next: float,
    dx: float,
    dy: float,
    no_slip_lat_ratio: float = NO_SLIP_LAT_RATIO,
) -> np.ndarray:
    c = math.cos(psi_next)
    s = math.sin(psi_next)
    long_comp = c * dx + s * dy
    lat_comp = -s * dx + c * dy

    lat_cap = no_slip_lat_ratio * max(abs(long_comp), 1e-3)
    lat_comp = float(np.clip(lat_comp, -lat_cap, lat_cap))

    dx_ns = long_comp * c - lat_comp * s
    dy_ns = long_comp * s + lat_comp * c
    return prev_xy + np.array([dx_ns, dy_ns], dtype=np.float32)


def _safe_distance_for_pair(type_i: int, type_j: int, len_i: float, len_j: float) -> float:
    if type_i == TYPE_VEHICLE and type_j == TYPE_VEHICLE:
        return max(2.0, 0.35 * (len_i + len_j) + 1.0)
    return 1.5


def apply_light_repulsion_step(
    step_states: np.ndarray,
    object_type: np.ndarray,
    size_step: np.ndarray,
    max_push: float | None = None,
    physics_cfg: dict | None = None,
) -> np.ndarray:
    cfg = resolve_physics_cfg(physics_cfg)
    if max_push is None:
        max_push = float(cfg["repulsion_max_push"])

    xy = step_states[:, :2].copy()
    n = xy.shape[0]
    for i in range(n):
        if int(object_type[i]) != TYPE_VEHICLE:
            continue
        for j in range(i + 1, n):
            if int(object_type[j]) != TYPE_VEHICLE:
                continue
            delta = xy[i] - xy[j]
            dist = float(np.linalg.norm(delta))
            safe_d = _safe_distance_for_pair(
                int(object_type[i]),
                int(object_type[j]),
                float(size_step[i, 0]),
                float(size_step[j, 0]),
            )
            if dist < 1e-4 or dist >= safe_d:
                continue
            direction = delta / dist
            push = min(float(max_push), 0.5 * (safe_d - dist))
            xy[i] += push * direction
            xy[j] -= push * direction

    out = step_states.copy()
    out[:, :2] = xy
    return out

def _decode_local_chunk_to_global_legacy(
    states: np.ndarray,
    pred_chunk_local: np.ndarray,
    object_type: np.ndarray,
    anchor_t: int,
    clamp_scale: float | None = 1.0,
    history_speed_bounds: np.ndarray | None = None,
) -> np.ndarray:
    n_obj = states.shape[0]
    out_states = states.copy()
    for f in range(pred_chunk_local.shape[1]):
        global_step = anchor_t + f + 1
        for i in range(n_obj):
            prev = out_states[i, global_step - 1]
            anchor = out_states[i, anchor_t]
            anchor_xy = anchor[:2]
            anchor_h = float(anchor[3])

            local_pred = pred_chunk_local[i, f]
            desired_xy = to_world(local_pred[:2], anchor_h) + anchor_xy
            desired_h = wrap_angle(anchor_h + math.atan2(float(local_pred[2]), float(local_pred[3])))
            proposed = np.array([desired_xy[0], desired_xy[1], anchor[2], desired_h], dtype=np.float32)

            if not np.all(np.isfinite(proposed)):
                out_states[i, global_step] = prev
                continue

            if clamp_scale is None:
                next_state = proposed
            else:
                speed_bounds_i = tuple(history_speed_bounds[i]) if history_speed_bounds is not None else None
                next_state = apply_type_aware_clamps(
                    prev,
                    proposed,
                    int(object_type[i]),
                    clamp_scale=float(clamp_scale),
                    speed_bounds=speed_bounds_i,
                )

            if not np.all(np.isfinite(next_state)):
                out_states[i, global_step] = prev
                continue

            out_states[i, global_step] = next_state
    return out_states


def _decode_local_chunk_to_global_bicycle(
    states: np.ndarray,
    pred_chunk_local: np.ndarray,
    object_type: np.ndarray,
    sizes: np.ndarray,
    valid: np.ndarray,
    map_cache: dict,
    anchor_t: int,
    dt: float,
    use_history_speed_clamp: bool,
    use_road_bound: bool,
    use_light_repulsion: bool,
    clamp_scale_for_non_vehicle: float | None,
    physics_cfg: dict | None = None,
) -> np.ndarray:
    cfg = resolve_physics_cfg(physics_cfg)
    accel_min = float(cfg["veh_accel_min"])
    accel_max = float(cfg["veh_accel_max"])
    steer_lim = float(cfg["veh_steer_limit"])
    no_slip_lat_ratio = float(cfg["no_slip_lat_ratio"])

    n_obj = states.shape[0]
    out_states = states.copy()

    init_dyn = estimate_vehicle_initial_dynamics(
        states=states,
        valid=valid,
        current_idx=anchor_t,
        object_type=object_type,
        sizes=sizes,
        dt=dt,
    )
    v_curr = init_dyn["v0"].copy()
    direction_sign = init_dyn["direction_sign"].copy()
    wheelbase = init_dyn["wheelbase"].copy()
    speed_bounds = init_dyn["speed_bounds"].copy()

    for f in range(pred_chunk_local.shape[1]):
        global_step = anchor_t + f + 1
        for i in range(n_obj):
            prev = out_states[i, global_step - 1]
            anchor = out_states[i, anchor_t]
            if not valid[i, anchor_t]:
                out_states[i, global_step] = prev
                continue

            anchor_xy = anchor[:2]
            anchor_h = float(anchor[3])
            local_pred = pred_chunk_local[i, f]
            desired_xy = to_world(local_pred[:2], anchor_h) + anchor_xy
            desired_h = wrap_angle(anchor_h + math.atan2(float(local_pred[2]), float(local_pred[3])))

            if int(object_type[i]) != TYPE_VEHICLE:
                proposed = np.array([desired_xy[0], desired_xy[1], anchor[2], desired_h], dtype=np.float32)
                if not np.all(np.isfinite(proposed)):
                    out_states[i, global_step] = prev
                    continue
                if clamp_scale_for_non_vehicle is None:
                    out_states[i, global_step] = proposed
                else:
                    out_states[i, global_step] = apply_type_aware_clamps(
                        prev,
                        proposed,
                        int(object_type[i]),
                        clamp_scale=float(clamp_scale_for_non_vehicle),
                    )
                continue

            a_cmd, delta_cmd = local_prediction_to_controls(
                current_state=prev,
                desired_xy=desired_xy,
                desired_heading=desired_h,
                v_curr=float(v_curr[i]),
                wheelbase=float(wheelbase[i]),
                dt=dt,
                physics_cfg=cfg,
            )

            a_cmd = float(np.clip(a_cmd, accel_min, accel_max))
            delta_cmd = float(np.clip(delta_cmd, -steer_lim, steer_lim))

            v_next = float(v_curr[i] + a_cmd * dt)
            if use_history_speed_clamp:
                vmin, vmax = speed_bounds[i]
                speed_abs = float(np.clip(abs(v_next), vmin, vmax))
                sign = 1.0 if v_next >= 0.0 else -1.0
                if abs(v_next) < 1e-3:
                    sign = float(direction_sign[i])
                v_next = sign * speed_abs

            yaw_rate = float((v_next / max(float(wheelbase[i]), 1e-3)) * math.tan(delta_cmd))
            yaw_rate_bound = float(abs(v_next) / max(float(wheelbase[i]), 1e-3) * math.tan(steer_lim) + 1e-4)
            yaw_rate = float(np.clip(yaw_rate, -yaw_rate_bound, yaw_rate_bound))
            psi_next = wrap_angle(float(prev[3]) + yaw_rate * dt)

            dx = float(v_next * math.cos(psi_next) * dt)
            dy = float(v_next * math.sin(psi_next) * dt)
            next_xy = _apply_no_slip_update(prev[:2], psi_next, dx, dy, no_slip_lat_ratio=no_slip_lat_ratio)

            if use_road_bound:
                next_xy = enforce_road_bound(next_xy, map_cache, physics_cfg=cfg)

            next_state = np.array([next_xy[0], next_xy[1], prev[2], psi_next], dtype=np.float32)
            if not np.all(np.isfinite(next_state)):
                next_state = prev

            out_states[i, global_step] = next_state
            v_curr[i] = v_next

        if use_light_repulsion:
            size_step = sizes[:, min(global_step, sizes.shape[1] - 1), :]
            out_states[:, global_step, :] = apply_light_repulsion_step(
                out_states[:, global_step, :],
                object_type=object_type,
                size_step=size_step,
                physics_cfg=cfg,
            )
            if use_road_bound:
                for i in range(n_obj):
                    if int(object_type[i]) == TYPE_VEHICLE:
                        out_states[i, global_step, :2] = enforce_road_bound(
                            out_states[i, global_step, :2],
                            map_cache,
                            physics_cfg=cfg,
                        )

    return out_states

def _guidance_score(pred_chunk_local: torch.Tensor, cond: dict) -> float:
    pred0 = pred_chunk_local[:, 0, :2]
    nbr_xy = cond["nbr"][..., :2]
    nbr_valid = cond["masks"]["nbr_valid"]
    d = torch.linalg.norm(pred0.unsqueeze(1) - nbr_xy, dim=-1)
    clearance = ((d - 2.0).clamp_min(0.0) * nbr_valid).sum() / nbr_valid.sum().clamp_min(1.0)
    lane_penalty = torch.linalg.norm(pred0, dim=-1).mean()
    return float((clearance - 0.1 * lane_penalty).item())


def _detect_high_speed_scene(states: np.ndarray, valid: np.ndarray, current_idx: int, dt: float = DT_DEFAULT) -> tuple[bool, float]:
    if current_idx <= 0:
        return False, 0.0

    prev_idx = max(0, current_idx - 1)
    valid_pair = valid[:, current_idx] & valid[:, prev_idx]
    if not np.any(valid_pair):
        return False, 0.0

    delta = states[valid_pair, current_idx, :2] - states[valid_pair, prev_idx, :2]
    speeds = np.linalg.norm(delta, axis=1) / max(dt, 1e-6)
    high_speed_frac = float(np.mean(speeds >= (HIGH_SPEED_STEP_THRESHOLD / max(dt, 1e-6))))
    return high_speed_frac >= 0.6, high_speed_frac


def predict_challenge_80(
    model: nn.Module,
    scene_features: dict,
    current_idx: int,
    guidance_values: list[float] | tuple[float, ...] = tuple(GUIDANCE_SWEEP),
    sample_steps: int = DIFFUSION_SAMPLE_STEPS,
    use_ema: bool = True,
    postprocess_mode: Literal["legacy", "bicycle"] = "bicycle",
    use_history_speed_clamp: bool = True,
    use_road_bound: bool = True,
    use_light_repulsion: bool = True,
    dt: float = DT_DEFAULT,
    use_clamping: bool = True,
    use_guidance_sweep: bool = True,
    sanitize_prediction: bool = True,
    force_highway_mode: bool | None = None,
    clamp_scale_override: float | None = None,
    fixed_guidance: float | None = None,
    inference_cfg: dict | None = None,
    physics_cfg: dict | None = None,
) -> tuple[np.ndarray, float, dict, bool, float]:
    if inference_cfg is not None:
        merged = resolve_inference_cfg(inference_cfg)
        guidance_values = tuple(merged["guidance_values"])
        sample_steps = int(merged["sample_steps"])
        use_ema = bool(merged["use_ema"])
        postprocess_mode = merged["postprocess_mode"]
        use_history_speed_clamp = bool(merged["use_history_speed_clamp"])
        use_road_bound = bool(merged["use_road_bound"])
        use_light_repulsion = bool(merged["use_light_repulsion"])
        dt = float(merged["dt"])
        use_clamping = bool(merged["use_clamping"])
        use_guidance_sweep = bool(merged["use_guidance_sweep"])
        sanitize_prediction = bool(merged["sanitize_prediction"])
        if merged.get("force_highway_mode") is not None:
            force_highway_mode = bool(merged.get("force_highway_mode"))
        if merged.get("clamp_scale_override") is not None:
            clamp_scale_override = float(merged.get("clamp_scale_override"))
        if merged.get("fixed_guidance") is not None:
            fixed_guidance = float(merged.get("fixed_guidance"))

    physics_cfg = resolve_physics_cfg(physics_cfg)

    base_states, base_sizes = sanitize_scene_arrays(
        scene_features["states"].copy(),
        scene_features["sizes"].copy(),
        scene_features["valid"],
    )
    valid = scene_features["valid"].copy()
    object_type = scene_features["object_type"]
    map_cache = scene_features["map_cache"]
    num_rollouts = int(scene_features.get("num_rollouts", 1))

    n_obj, orig_n_steps, _ = base_states.shape
    end_step = current_idx + FUTURE_STEPS
    ext_len = max(orig_n_steps, end_step + 1)

    ext_states = np.zeros((n_obj, ext_len, 4), dtype=np.float32)
    ext_states[:, :orig_n_steps, :] = base_states
    ext_valid = np.zeros((n_obj, ext_len), dtype=bool)
    ext_valid[:, :orig_n_steps] = valid

    ext_sizes = np.zeros((n_obj, ext_len, 3), dtype=np.float32)
    ext_sizes[:, :orig_n_steps, :] = base_sizes

    step_idx = np.tile(np.arange(orig_n_steps)[None, :], (n_obj, 1))
    last_valid_idx = (valid.astype(np.int32) * step_idx).max(axis=1)
    for i in range(n_obj):
        if not ext_valid[i, current_idx]:
            continue
        last_known = int(last_valid_idx[i])
        ext_states[i, last_known + 1 :, :] = ext_states[i, last_known:last_known + 1, :]
        ext_valid[i, last_known + 1 :] = True
        ext_sizes[i, orig_n_steps:, :] = ext_sizes[i, last_known:last_known + 1, :]

    highway_mode, high_speed_frac = _detect_high_speed_scene(ext_states, ext_valid, current_idx, dt=dt)
    if force_highway_mode is not None:
        highway_mode = bool(force_highway_mode)

    if use_guidance_sweep and highway_mode:
        guidance_values = tuple(HIGHWAY_GUIDANCE_SWEEP)
        sample_steps = max(int(sample_steps), int(HIGHWAY_SAMPLE_STEPS))

    cond = build_rollout_condition_batch(
        states=ext_states,
        valid=ext_valid,
        object_type=object_type,
        sizes=ext_sizes,
        map_cache=map_cache,
        current_t=current_idx,
        H=HISTORY_STEPS,
        K=NEIGHBORS_K,
    )

    if use_guidance_sweep:
        scores = {}
        for g in guidance_values:
            cfg = dict(DIFFUSION_CFG)
            cfg["guidance_scale"] = float(g)
            pilot_pred = sample_future_chunk(model, cond, diffusion_cfg=cfg, use_ema=use_ema, sample_steps=sample_steps)
            scores[float(g)] = _guidance_score(pilot_pred, cond)
        chosen_guidance = max(scores, key=scores.get)
    else:
        chosen_guidance = float(
            fixed_guidance
            if fixed_guidance is not None
            else (guidance_values[0] if len(guidance_values) > 0 else DIFFUSION_CFG["guidance_scale"])
        )
        scores = {float(chosen_guidance): 0.0}

    if clamp_scale_override is not None:
        clamp_scale = float(clamp_scale_override)
    elif use_clamping:
        clamp_scale = (HIGHWAY_CLAMP_SCALE if highway_mode else 1.0)
    else:
        clamp_scale = None

    init_dyn = estimate_vehicle_initial_dynamics(
        states=ext_states,
        valid=ext_valid,
        current_idx=current_idx,
        object_type=object_type,
        sizes=ext_sizes,
        dt=dt,
    )
    history_speed_bounds = init_dyn["speed_bounds"] if use_history_speed_clamp else None

    all_rollouts = np.zeros((num_rollouts, n_obj, ext_len, 4), dtype=np.float32)

    for r in range(num_rollouts):
        cfg = dict(DIFFUSION_CFG)
        cfg["guidance_scale"] = float(chosen_guidance)
        pred_chunk_local = sample_future_chunk(
            model,
            cond,
            diffusion_cfg=cfg,
            use_ema=use_ema,
            sample_steps=sample_steps,
        ).detach().cpu().numpy()

        if sanitize_prediction:
            pred_chunk_local = np.nan_to_num(pred_chunk_local, nan=0.0, posinf=0.0, neginf=0.0)

        rollout_states = ext_states.copy()
        rollout_states[:, current_idx + 1 :, :] = rollout_states[:, current_idx:current_idx + 1, :]

        if postprocess_mode == "legacy":
            rollout_states = _decode_local_chunk_to_global_legacy(
                rollout_states,
                pred_chunk_local,
                object_type=object_type,
                anchor_t=current_idx,
                clamp_scale=clamp_scale,
                history_speed_bounds=history_speed_bounds,
            )
        elif postprocess_mode == "bicycle":
            rollout_states = _decode_local_chunk_to_global_bicycle(
                rollout_states,
                pred_chunk_local,
                object_type=object_type,
                sizes=ext_sizes,
                valid=ext_valid,
                map_cache=map_cache,
                anchor_t=current_idx,
                dt=dt,
                use_history_speed_clamp=use_history_speed_clamp,
                use_road_bound=use_road_bound,
                use_light_repulsion=use_light_repulsion,
                clamp_scale_for_non_vehicle=clamp_scale,
                physics_cfg=physics_cfg,
            )
        else:
            raise ValueError(f"Unsupported postprocess_mode={postprocess_mode}")

        all_rollouts[r] = rollout_states

    return all_rollouts, float(chosen_guidance), scores, highway_mode, high_speed_frac

def compute_postprocess_metrics(
    simulated_states: np.ndarray,
    scene_features: dict,
    current_idx: int,
    dt: float = DT_DEFAULT,
    physics_cfg: dict | None = None,
) -> dict:
    cfg = resolve_physics_cfg(physics_cfg)
    min_edge_clearance = float(cfg["min_edge_clearance_m"])

    states = simulated_states[0]
    n_obj, n_steps, _ = states.shape
    object_type = scene_features["object_type"]
    sizes = np.nan_to_num(scene_features["sizes"], nan=0.0, posinf=0.0, neginf=0.0)
    map_cache = scene_features["map_cache"]

    vehicle_mask = (object_type == TYPE_VEHICLE)
    start = int(current_idx + 1)
    end = min(n_steps, current_idx + FUTURE_STEPS + 1)

    nan_count = int(np.isnan(states[:, start:end]).sum())

    slip_vals = []
    for i in range(n_obj):
        if not vehicle_mask[i]:
            continue
        for t in range(start + 1, end):
            dxy = states[i, t, :2] - states[i, t - 1, :2]
            psi = float(states[i, t - 1, 3])
            c, s = math.cos(psi), math.sin(psi)
            v_long = (c * dxy[0] + s * dxy[1]) / max(dt, 1e-6)
            v_lat = (-s * dxy[0] + c * dxy[1]) / max(dt, 1e-6)
            if abs(v_long) > 0.2:
                slip_vals.append(abs(v_lat) / abs(v_long))
    mean_slip = float(np.mean(slip_vals)) if slip_vals else 0.0

    edge_pts = map_cache.get("road_edge_points", np.zeros((0, 2), dtype=np.float32))
    offroad_violations = 0
    if edge_pts.shape[0] > 0:
        for i in range(n_obj):
            if not vehicle_mask[i]:
                continue
            for t in range(start, end):
                d, _ = _distance_to_nearest(edge_pts, states[i, t, :2])
                if d < min_edge_clearance:
                    offroad_violations += 1

    collision_pairs = 0
    for t in range(start, end):
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                pi = states[i, t, :2]
                pj = states[j, t, :2]
                d = float(np.linalg.norm(pi - pj))
                len_i = float(sizes[i, min(t, sizes.shape[1] - 1), 0])
                len_j = float(sizes[j, min(t, sizes.shape[1] - 1), 0])
                safe_d = _safe_distance_for_pair(int(object_type[i]), int(object_type[j]), len_i, len_j)
                if d < max(0.8, 0.7 * safe_d):
                    collision_pairs += 1

    return {
        "nan_count": nan_count,
        "mean_vehicle_slip_ratio": mean_slip,
        "offroad_violations": int(offroad_violations),
        "collision_pairs": int(collision_pairs),
    }

def run_postprocess_ablation(
    model: nn.Module,
    scene_features: dict,
    current_idx: int,
    use_ema: bool = True,
    physics_cfg: dict | None = None,
) -> dict:
    configs = {
        "legacy": dict(postprocess_mode="legacy", use_road_bound=False, use_light_repulsion=False),
        "bicycle_no_safety": dict(postprocess_mode="bicycle", use_road_bound=False, use_light_repulsion=False),
        "bicycle_full": dict(postprocess_mode="bicycle", use_road_bound=True, use_light_repulsion=True),
    }

    results = {}
    for name, cfg in configs.items():
        t0 = time.perf_counter()
        sim, chosen_g, scores, highway_mode, high_speed_frac = predict_challenge_80(
            model=model,
            scene_features=scene_features,
            current_idx=current_idx,
            guidance_values=(1.0,),
            sample_steps=DIFFUSION_SAMPLE_STEPS,
            use_ema=use_ema,
            use_guidance_sweep=False,
            sanitize_prediction=True,
            force_highway_mode=False,
            use_history_speed_clamp=True,
            dt=DT_DEFAULT,
            physics_cfg=physics_cfg,
            **cfg,
        )
        elapsed = time.perf_counter() - t0
        metrics = compute_postprocess_metrics(
            sim,
            scene_features,
            current_idx=current_idx,
            dt=DT_DEFAULT,
            physics_cfg=physics_cfg,
        )
        results[name] = {
            "runtime_s": float(elapsed),
            "chosen_guidance": float(chosen_g),
            "guidance_scores": scores,
            "highway_mode": bool(highway_mode),
            "high_speed_frac": float(high_speed_frac),
            **metrics,
        }

    print("Ablation summary:")
    for name, item in results.items():
        print(
            f"{name:18s} | runtime={item['runtime_s']:.3f}s | "
            f"offroad={item['offroad_violations']} | collisions={item['collision_pairs']} | "
            f"slip={item['mean_vehicle_slip_ratio']:.4f}"
        )

    return results


def list_validation_tfrecords(gcs_val_path: str = GCS_VAL_PATH, refresh: bool = False) -> list[str]:
    global VAL_FILE_CACHE
    if VAL_FILE_CACHE is not None and not refresh:
        return VAL_FILE_CACHE
    result = subprocess.run(["gsutil", "ls", gcs_val_path], capture_output=True, text=True, check=True)
    VAL_FILE_CACHE = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return VAL_FILE_CACHE


def sample_cross_file_scenarios(
    n: int = 10,
    seed: int = 2026,
    cleanup_local: bool = False,
    cache_dir: str = LOCAL_WAYMO_CACHE_DIR,
    force_refresh: bool = False,
    verbose: bool = False,
) -> list[dict]:
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

def build_scene_features_for_eda(
    scenario_proto: scenario_pb2.Scenario,
    challenge_type,
    submission_config,
) -> dict:
    scene_features = build_scene_rollout_features(scenario_proto, challenge_type, submission_config)
    scene_features["num_rollouts"] = 1
    return scene_features


def rollout_cv_baseline(
    scene_features: dict,
    current_idx: int,
    horizon: int = FUTURE_STEPS,
    dt: float = DT_DEFAULT,
) -> np.ndarray:
    base_states, base_sizes = sanitize_scene_arrays(
        scene_features["states"].copy(),
        scene_features["sizes"].copy(),
        scene_features["valid"],
    )
    valid = scene_features["valid"].copy()

    n_obj, orig_n_steps, _ = base_states.shape
    end_step = current_idx + int(horizon)
    ext_len = max(orig_n_steps, end_step + 1)

    ext_states = np.zeros((n_obj, ext_len, 4), dtype=np.float32)
    ext_states[:, :orig_n_steps, :] = base_states
    ext_valid = np.zeros((n_obj, ext_len), dtype=bool)
    ext_valid[:, :orig_n_steps] = valid

    ext_sizes = np.zeros((n_obj, ext_len, 3), dtype=np.float32)
    ext_sizes[:, :orig_n_steps, :] = base_sizes

    step_idx = np.tile(np.arange(orig_n_steps)[None, :], (n_obj, 1))
    last_valid_idx = (valid.astype(np.int32) * step_idx).max(axis=1)
    for i in range(n_obj):
        if not ext_valid[i, current_idx]:
            continue
        last_known = int(last_valid_idx[i])
        ext_states[i, last_known + 1 :, :] = ext_states[i, last_known:last_known + 1, :]
        ext_valid[i, last_known + 1 :] = True
        ext_sizes[i, orig_n_steps:, :] = ext_sizes[i, last_known:last_known + 1, :]

    for i in range(n_obj):
        if not ext_valid[i, current_idx]:
            continue

        prev_idx = max(0, current_idx - 1)
        has_prev = ext_valid[i, prev_idx]

        if has_prev:
            step_delta = ext_states[i, current_idx, :2] - ext_states[i, prev_idx, :2]
            heading_delta = wrap_angle(float(ext_states[i, current_idx, 3]) - float(ext_states[i, prev_idx, 3]))
        else:
            step_delta = np.zeros((2,), dtype=np.float32)
            heading_delta = 0.0

        for t in range(current_idx + 1, end_step + 1):
            prev = ext_states[i, t - 1]
            next_xy = prev[:2] + step_delta
            next_h = wrap_angle(float(prev[3]) + float(heading_delta))
            ext_states[i, t] = np.array([next_xy[0], next_xy[1], prev[2], next_h], dtype=np.float32)

    return ext_states[None, ...]


def compute_error_metrics(
    pred_states: np.ndarray,
    gt_states: np.ndarray,
    gt_valid: np.ndarray,
    current_idx: int,
    horizon: int = FUTURE_STEPS,
) -> dict:
    start = int(current_idx + 1)
    end = min(start + int(horizon), pred_states.shape[1], gt_states.shape[1])

    if end <= start:
        return {
            "step1_l2": float("inf"),
            "ade80": float("inf"),
            "fde80": float("inf"),
            "valid_future_points": 0,
        }

    pred_xy = pred_states[:, start:end, :2]
    gt_xy = gt_states[:, start:end, :2]
    valid_future = gt_valid[:, start:end]
    dists = np.linalg.norm(pred_xy - gt_xy, axis=-1)

    if np.any(valid_future):
        ade = float(np.mean(dists[valid_future]))
    else:
        ade = float("inf")

    step1_l2 = float("inf")
    if start < gt_valid.shape[1]:
        valid_step1 = gt_valid[:, start]
        if np.any(valid_step1):
            step1_l2 = float(np.mean(dists[:, 0][valid_step1]))

    fde = float("inf")
    fde_idx_local = dists.shape[1] - 1
    fde_idx_global = start + fde_idx_local
    if fde_idx_global < gt_valid.shape[1]:
        valid_fde = gt_valid[:, fde_idx_global]
        if np.any(valid_fde):
            fde = float(np.mean(dists[:, fde_idx_local][valid_fde]))

    return {
        "step1_l2": step1_l2,
        "ade80": ade,
        "fde80": fde,
        "valid_future_points": int(valid_future.sum()),
    }


def compute_scene_score(model_metrics: dict, cv_metrics: dict) -> dict:
    gain_step1 = (cv_metrics["step1_l2"] - model_metrics["step1_l2"]) / max(cv_metrics["step1_l2"], 1e-6)
    gain_ade80 = (cv_metrics["ade80"] - model_metrics["ade80"]) / max(cv_metrics["ade80"], 1e-6)
    gain_fde80 = (cv_metrics["fde80"] - model_metrics["fde80"]) / max(cv_metrics["fde80"], 1e-6)
    gain_offroad = (
        cv_metrics["offroad_violations"] - model_metrics["offroad_violations"]
    ) / max(float(cv_metrics["offroad_violations"]), 1.0)
    gain_collision = (
        cv_metrics["collision_pairs"] - model_metrics["collision_pairs"]
    ) / max(float(cv_metrics["collision_pairs"]), 1.0)

    final_score = (
        0.35 * gain_step1
        + 0.30 * gain_ade80
        + 0.15 * gain_fde80
        + 0.10 * gain_offroad
        + 0.10 * gain_collision
    )

    return {
        "gain_step1": float(gain_step1),
        "gain_ade80": float(gain_ade80),
        "gain_fde80": float(gain_fde80),
        "gain_offroad": float(gain_offroad),
        "gain_collision": float(gain_collision),
        "final_score": float(final_score),
    }


def default_eda_config(config_id: str = "default") -> dict:
    return {
        "config_id": config_id,
        "inference_cfg": dict(DEFAULT_INFERENCE_CFG),
        "physics_cfg": dict(DEFAULT_PHYSICS_CFG),
    }

def sample_random_eda_configs(n_configs: int = 24, seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs = []
    sample_steps_grid = [32, 40, 50, 64]

    for i in range(int(n_configs)):
        guidance_scale = float(np.round(rng.uniform(1.0, 1.8), 3))
        steer_lim = float(np.round(rng.uniform(0.40, 0.60), 3))

        inf_cfg = {
            "guidance_values": (guidance_scale,),
            "sample_steps": int(rng.choice(sample_steps_grid)),
            "use_ema": True,
            "postprocess_mode": "bicycle",
            "use_history_speed_clamp": bool(rng.integers(0, 2)),
            "use_road_bound": bool(rng.integers(0, 2)),
            "use_light_repulsion": bool(rng.integers(0, 2)),
            "dt": DT_DEFAULT,
            "use_clamping": True,
            "use_guidance_sweep": False,
            "sanitize_prediction": True,
            "force_highway_mode": False,
            "fixed_guidance": guidance_scale,
        }
        phy_cfg = {
            "veh_accel_min": float(np.round(rng.uniform(-4.5, -2.5), 3)),
            "veh_accel_max": float(np.round(rng.uniform(1.8, 3.2), 3)),
            "veh_steer_limit": steer_lim,
            "no_slip_lat_ratio": float(np.round(rng.uniform(0.02, 0.08), 4)),
            "min_edge_clearance_m": float(np.round(rng.uniform(0.8, 1.8), 3)),
            "lane_pull_gain": float(np.round(rng.uniform(0.08, 0.30), 3)),
            "repulsion_max_push": float(np.round(rng.uniform(0.15, 0.55), 3)),
        }
        configs.append({
            "config_id": f"rand_{i:02d}",
            "inference_cfg": inf_cfg,
            "physics_cfg": phy_cfg,
        })

    return configs


def _evaluate_single_config(
    model: nn.Module,
    config: dict,
    scenario_entries: list[dict],
    submission_config,
    challenge_type,
    seed_list: list[int],
    stage_label: str,
) -> list[dict]:
    rows = []
    cfg_id = config["config_id"]
    inference_cfg = resolve_inference_cfg(config["inference_cfg"])
    physics_cfg = resolve_physics_cfg(config["physics_cfg"])

    for entry in scenario_entries:
        scene_features = entry["scene_features"]
        current_idx = int(submission_config.current_time_index)

        gt_states = scene_features["states"]
        gt_valid = scene_features["valid"]
        assert np.isfinite(gt_states).all(), "Non-finite ground-truth states in sampled scenario"

        cv_rollout = rollout_cv_baseline(scene_features, current_idx=current_idx, horizon=FUTURE_STEPS, dt=inference_cfg["dt"])
        cv_post = compute_postprocess_metrics(
            cv_rollout,
            scene_features,
            current_idx=current_idx,
            dt=inference_cfg["dt"],
            physics_cfg=physics_cfg,
        )
        cv_err = compute_error_metrics(
            cv_rollout[0],
            gt_states,
            gt_valid,
            current_idx=current_idx,
            horizon=FUTURE_STEPS,
        )
        cv_metrics = {
            "step1_l2": cv_err["step1_l2"],
            "ade80": cv_err["ade80"],
            "fde80": cv_err["fde80"],
            "offroad_violations": cv_post["offroad_violations"],
            "collision_pairs": cv_post["collision_pairs"],
        }

        for seed in seed_list:
            seed_everything(int(seed))
            t0 = time.perf_counter()
            sim, chosen_guidance, _, highway_mode, high_speed_frac = predict_challenge_80(
                model=model,
                scene_features=scene_features,
                current_idx=current_idx,
                inference_cfg=inference_cfg,
                physics_cfg=physics_cfg,
            )
            runtime_s = float(time.perf_counter() - t0)

            model_post = compute_postprocess_metrics(
                sim,
                scene_features,
                current_idx=current_idx,
                dt=inference_cfg["dt"],
                physics_cfg=physics_cfg,
            )
            model_err = compute_error_metrics(
                sim[0],
                gt_states,
                gt_valid,
                current_idx=current_idx,
                horizon=FUTURE_STEPS,
            )
            model_metrics = {
                "step1_l2": model_err["step1_l2"],
                "ade80": model_err["ade80"],
                "fde80": model_err["fde80"],
                "offroad_violations": model_post["offroad_violations"],
                "collision_pairs": model_post["collision_pairs"],
            }
            score = compute_scene_score(model_metrics, cv_metrics)

            finite_check_vals = [
                model_err["step1_l2"], model_err["ade80"], model_err["fde80"],
                cv_err["step1_l2"], cv_err["ade80"], cv_err["fde80"],
                score["final_score"],
            ]
            assert np.isfinite(finite_check_vals).all(), "Non-finite EDA metric encountered"

            row = {
                "stage": stage_label,
                "config_id": cfg_id,
                "scenario_uid": f"f{entry['file_index']:03d}_s{entry['scenario_index']:05d}_{entry['scenario_id']}",
                "scenario_id": entry["scenario_id"],
                "file_index": int(entry["file_index"]),
                "scenario_index": int(entry["scenario_index"]),
                "seed": int(seed),
                "runtime_s": runtime_s,
                "chosen_guidance": float(chosen_guidance),
                "highway_mode": bool(highway_mode),
                "high_speed_frac": float(high_speed_frac),
                "guidance_scale": float(inference_cfg.get("fixed_guidance", inference_cfg["guidance_values"][0])),
                "sample_steps": int(inference_cfg["sample_steps"]),
                "use_history_speed_clamp": int(bool(inference_cfg["use_history_speed_clamp"])),
                "use_road_bound": int(bool(inference_cfg["use_road_bound"])),
                "use_light_repulsion": int(bool(inference_cfg["use_light_repulsion"])),
                "veh_accel_min": float(physics_cfg["veh_accel_min"]),
                "veh_accel_max": float(physics_cfg["veh_accel_max"]),
                "veh_steer_limit": float(physics_cfg["veh_steer_limit"]),
                "no_slip_lat_ratio": float(physics_cfg["no_slip_lat_ratio"]),
                "min_edge_clearance_m": float(physics_cfg["min_edge_clearance_m"]),
                "lane_pull_gain": float(physics_cfg["lane_pull_gain"]),
                "repulsion_max_push": float(physics_cfg["repulsion_max_push"]),
                "model_step1_l2": float(model_err["step1_l2"]),
                "model_ade80": float(model_err["ade80"]),
                "model_fde80": float(model_err["fde80"]),
                "model_offroad_violations": int(model_post["offroad_violations"]),
                "model_collision_pairs": int(model_post["collision_pairs"]),
                "model_mean_vehicle_slip_ratio": float(model_post["mean_vehicle_slip_ratio"]),
                "cv_step1_l2": float(cv_err["step1_l2"]),
                "cv_ade80": float(cv_err["ade80"]),
                "cv_fde80": float(cv_err["fde80"]),
                "cv_offroad_violations": int(cv_post["offroad_violations"]),
                "cv_collision_pairs": int(cv_post["collision_pairs"]),
                **score,
            }
            rows.append(row)

    return rows


def aggregate_eda_runs(runs_df):
    if pd is None:
        raise ImportError("pandas is required for EDA aggregation")
    if runs_df.empty:
        return runs_df.copy(), runs_df.copy()

    metric_cols = [
        "runtime_s",
        "model_step1_l2",
        "model_ade80",
        "model_fde80",
        "model_offroad_violations",
        "model_collision_pairs",
        "model_mean_vehicle_slip_ratio",
        "gain_step1",
        "gain_ade80",
        "gain_fde80",
        "gain_offroad",
        "gain_collision",
        "final_score",
    ]

    scene_medians = (
        runs_df.groupby(["config_id", "scenario_uid"], as_index=False)[metric_cols]
        .median(numeric_only=True)
    )

    summary_df = (
        scene_medians.groupby("config_id", as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )

    param_cols = [
        "guidance_scale",
        "sample_steps",
        "use_history_speed_clamp",
        "use_road_bound",
        "use_light_repulsion",
        "veh_accel_min",
        "veh_accel_max",
        "veh_steer_limit",
        "no_slip_lat_ratio",
        "min_edge_clearance_m",
        "lane_pull_gain",
        "repulsion_max_push",
    ]
    param_df = runs_df.groupby("config_id", as_index=False)[param_cols].first()
    summary_df = summary_df.merge(param_df, on="config_id", how="left")

    summary_df = summary_df.sort_values(
        ["final_score", "model_step1_l2", "model_offroad_violations"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return summary_df, scene_medians


def plot_eda_param_vs_score(summary_df):
    if summary_df is None or len(summary_df) == 0:
        print("No summary rows to plot.")
        return

    params = [
        "guidance_scale",
        "sample_steps",
        "veh_accel_min",
        "veh_accel_max",
        "veh_steer_limit",
        "no_slip_lat_ratio",
        "min_edge_clearance_m",
        "lane_pull_gain",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, p in zip(axes, params):
        ax.scatter(summary_df[p], summary_df["final_score"], alpha=0.8)
        ax.set_title(f"{p} vs final_score")
        ax.set_xlabel(p)
        ax.set_ylabel("final_score")
        ax.grid(True, alpha=0.2)

    for i in range(len(params), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def compare_default_vs_best(summary_df, default_id: str = "default"):
    if summary_df is None or len(summary_df) == 0:
        return None

    best = summary_df.iloc[0]
    default_rows = summary_df[summary_df["config_id"] == default_id]
    if len(default_rows) == 0:
        return None
    default = default_rows.iloc[0]

    if pd is None:
        return None

    cols = [
        "final_score",
        "model_step1_l2",
        "model_ade80",
        "model_fde80",
        "model_offroad_violations",
        "model_collision_pairs",
        "runtime_s",
    ]
    cmp_df = pd.DataFrame(
        [default[cols].to_dict(), best[cols].to_dict()],
        index=["default", "best"],
    )
    return cmp_df


def _score_directionality_sanity_check() -> None:
    cv = {
        "step1_l2": 2.0,
        "ade80": 4.0,
        "fde80": 6.0,
        "offroad_violations": 10,
        "collision_pairs": 8,
    }
    better = {
        "step1_l2": 1.0,
        "ade80": 2.0,
        "fde80": 3.0,
        "offroad_violations": 5,
        "collision_pairs": 4,
    }
    worse = {
        "step1_l2": 2.5,
        "ade80": 5.0,
        "fde80": 7.0,
        "offroad_violations": 12,
        "collision_pairs": 9,
    }
    s_better = compute_scene_score(better, cv)["final_score"]
    s_worse = compute_scene_score(worse, cv)["final_score"]
    assert s_better > s_worse, "Scene score directionality sanity check failed"


def run_eda_hyperparam_search(
    model: nn.Module,
    n_scenarios: int = 10,
    scenario_seed: int = 2026,
    stage_a_num_configs: int = 24,
    stage_a_seeds: tuple[int, ...] = (11, 29),
    stage_b_top_k: int = 3,
    stage_b_seeds: tuple[int, ...] = (101, 127, 151, 181, 211, 239, 269, 307, 347, 389),
    verify_stage_a_repro: bool = False,
) -> dict:
    if pd is None:
        raise ImportError("pandas is required for EDA sweep. Install pandas in this notebook env.")

    _score_directionality_sanity_check()

    sampled = sample_cross_file_scenarios(
        n=n_scenarios,
        seed=scenario_seed,
        cleanup_local=False,
        cache_dir=LOCAL_WAYMO_CACHE_DIR,
        force_refresh=False,
        verbose=False,
    )
    assert len(sampled) == min(n_scenarios, len(list_validation_tfrecords())), "Unexpected sampled scenario count"

    challenge_type = submission_specs.ChallengeType.SCENARIO_GEN
    submission_config = submission_specs.get_submission_config(challenge_type)

    prepared_scenarios = []
    for entry in sampled:
        scene_features = build_scene_features_for_eda(entry["scenario_proto"], challenge_type, submission_config)
        assert np.isfinite(scene_features["states"]).all(), "Non-finite states in sampled scene_features"
        prepared_scenarios.append({**entry, "scene_features": scene_features})

    random_cfgs = sample_random_eda_configs(n_configs=stage_a_num_configs, seed=scenario_seed + 17)

    stage_a_rows = []
    for cfg in random_cfgs:
        stage_a_rows.extend(
            _evaluate_single_config(
                model=model,
                config=cfg,
                scenario_entries=prepared_scenarios,
                submission_config=submission_config,
                challenge_type=challenge_type,
                seed_list=list(stage_a_seeds),
                stage_label="stage_a",
            )
        )

    runs_a_df = pd.DataFrame(stage_a_rows)
    summary_a_df, scene_medians_a_df = aggregate_eda_runs(runs_a_df)

    repro_max_abs_diff = None
    if verify_stage_a_repro:
        stage_a_rows_repeat = []
        for cfg in random_cfgs:
            stage_a_rows_repeat.extend(
                _evaluate_single_config(
                    model=model,
                    config=cfg,
                    scenario_entries=prepared_scenarios,
                    submission_config=submission_config,
                    challenge_type=challenge_type,
                    seed_list=list(stage_a_seeds),
                    stage_label="stage_a_repeat",
                )
            )
        repeat_df = pd.DataFrame(stage_a_rows_repeat)
        repeat_summary_df, _ = aggregate_eda_runs(repeat_df)
        merged = summary_a_df[["config_id", "final_score"]].merge(
            repeat_summary_df[["config_id", "final_score"]],
            on="config_id",
            suffixes=("_a", "_b"),
        )
        repro_max_abs_diff = float(np.max(np.abs(merged["final_score_a"] - merged["final_score_b"])))

    top_ids = summary_a_df["config_id"].head(int(stage_b_top_k)).tolist()
    top_cfgs = [cfg for cfg in random_cfgs if cfg["config_id"] in top_ids]

    baseline_cfg = default_eda_config(config_id="default")
    stage_b_cfgs = top_cfgs + [baseline_cfg]

    stage_b_rows = []
    for cfg in stage_b_cfgs:
        stage_b_rows.extend(
            _evaluate_single_config(
                model=model,
                config=cfg,
                scenario_entries=prepared_scenarios,
                submission_config=submission_config,
                challenge_type=challenge_type,
                seed_list=list(stage_b_seeds),
                stage_label="stage_b",
            )
        )

    runs_b_df = pd.DataFrame(stage_b_rows)
    summary_b_df, scene_medians_b_df = aggregate_eda_runs(runs_b_df)

    runs_df = pd.concat([runs_a_df, runs_b_df], ignore_index=True)

    summary_df = summary_b_df.copy()
    score_order_cols = ["final_score", "model_step1_l2", "model_offroad_violations"]
    summary_df = summary_df.sort_values(score_order_cols, ascending=[False, True, True]).reset_index(drop=True)

    config_registry = {cfg["config_id"]: cfg for cfg in stage_b_cfgs}
    default_rows = summary_df[summary_df["config_id"] == "default"]

    selection_reason = "score-only fallback"
    if len(default_rows) > 0:
        default_row = default_rows.iloc[0]
        ade_gate = float(default_row["model_ade80"])
        offroad_gate = float(default_row["model_offroad_violations"]) * 1.05
        collision_gate = float(default_row["model_collision_pairs"]) * 1.05

        summary_df["passes_safety_gate"] = (
            (summary_df["model_ade80"] <= ade_gate)
            & (summary_df["model_offroad_violations"] <= offroad_gate)
            & (summary_df["model_collision_pairs"] <= collision_gate)
        ).astype(int)

        gated_df = summary_df[summary_df["passes_safety_gate"] == 1].copy()
        gated_df = gated_df.sort_values(score_order_cols, ascending=[False, True, True]).reset_index(drop=True)

        if len(gated_df) > 0:
            selected_row = gated_df.iloc[0]
            selection_reason = f"safety-gated pick ({len(gated_df)} configs passed)"
        else:
            selected_row = default_row
            selection_reason = "no config passed safety gate; fallback to default"
    else:
        summary_df["passes_safety_gate"] = 0
        selected_row = summary_df.iloc[0]
        selection_reason = "default config not found in stage-B summary"

    top10 = summary_df.head(10)
    print("Top-10 Stage-B configs:")
    print(top10[[
        "config_id",
        "passes_safety_gate",
        "final_score",
        "model_step1_l2",
        "model_ade80",
        "model_fde80",
        "model_offroad_violations",
        "model_collision_pairs",
        "runtime_s",
    ]])

    plot_eda_param_vs_score(summary_df)

    selected_id = str(selected_row["config_id"])
    selected_cfg = config_registry[selected_id]
    selected_score = float(selected_row["final_score"])

    print(f"\nSelected config: {selected_id}")
    print("Selection reason:", selection_reason)

    default_row = summary_df[summary_df["config_id"] == "default"]
    if len(default_row) > 0:
        default_row = default_row.iloc[0]
        default_score = float(default_row["final_score"])
        default_step1 = float(default_row["model_step1_l2"])
        default_ade = float(default_row["model_ade80"])
        default_offroad = float(default_row["model_offroad_violations"])
        default_collision = float(default_row["model_collision_pairs"])

        selected_step1 = float(selected_row["model_step1_l2"])
        selected_ade = float(selected_row["model_ade80"])
        selected_offroad = float(selected_row["model_offroad_violations"])
        selected_collision = float(selected_row["model_collision_pairs"])

        cmp_df = pd.DataFrame(
            [
                {
                    "final_score": default_score,
                    "model_step1_l2": default_step1,
                    "model_ade80": default_ade,
                    "model_fde80": float(default_row["model_fde80"]),
                    "model_offroad_violations": default_offroad,
                    "model_collision_pairs": default_collision,
                    "runtime_s": float(default_row["runtime_s"]),
                },
                {
                    "final_score": selected_score,
                    "model_step1_l2": selected_step1,
                    "model_ade80": selected_ade,
                    "model_fde80": float(selected_row["model_fde80"]),
                    "model_offroad_violations": selected_offroad,
                    "model_collision_pairs": selected_collision,
                    "runtime_s": float(selected_row["runtime_s"]),
                },
            ],
            index=["default", "selected"],
        )
        print("\nDefault vs Selected (Stage-B aggregate):")
        print(cmp_df)

        print(f"\nFinal score improvement vs default: {selected_score - default_score:+.4f}")
        print("Acceptance checks:")
        print("- final_score improved:", selected_score > default_score)
        print("- step1_l2 improved:", selected_step1 < default_step1)
        print("- ade80 improved:", selected_ade < default_ade)
        print("- offroad regression <= 5%:", selected_offroad <= default_offroad * 1.05)
        print("- collision regression <= 5%:", selected_collision <= default_collision * 1.05)

    median_runtime = float(runs_df["runtime_s"].median())
    total_runs = len(runs_df)
    est_hours = float((median_runtime * total_runs) / 3600.0)
    print(f"Median runtime/run: {median_runtime:.3f}s | total runs: {total_runs} | estimated wall-clock: {est_hours:.2f}h")

    BEST_CFG = {
        "inference_cfg": selected_cfg["inference_cfg"],
        "physics_cfg": selected_cfg["physics_cfg"],
    }
    print("\nBEST_CFG (paste-ready):")
    print(json.dumps(BEST_CFG, indent=2))

    return {
        "runs_df": runs_df,
        "summary_df": summary_df,
        "stage_a_summary_df": summary_a_df,
        "stage_a_scene_medians_df": scene_medians_a_df,
        "stage_b_scene_medians_df": scene_medians_b_df,
        "sampled_scenarios": sampled,
        "BEST_CFG": BEST_CFG,
        "selected_config_id": selected_id,
        "selection_reason": selection_reason,
        "repro_max_abs_diff": repro_max_abs_diff,
    }

def rollout_in_chunks(
    model: nn.Module,
    scene_features: dict,
    current_idx: int,
    horizon: int,
    F: int = FUTURE_STEPS,
    use_ema: bool = True,
) -> np.ndarray:
    base_states, base_sizes = sanitize_scene_arrays(
        scene_features["states"].copy(),
        scene_features["sizes"].copy(),
        scene_features["valid"],
    )
    valid = scene_features["valid"].copy()
    object_type = scene_features["object_type"]
    map_cache = scene_features["map_cache"]

    num_rollouts = scene_features.get("num_rollouts", 1)

    n_obj, orig_n_steps, _ = base_states.shape
    ext_len = max(orig_n_steps, current_idx + horizon + 1)
    end_step = current_idx + horizon

    ext_states = np.zeros((n_obj, ext_len, 4), dtype=np.float32)
    ext_states[:, :orig_n_steps, :] = base_states

    ext_valid = np.zeros((n_obj, ext_len), dtype=bool)
    ext_valid[:, :orig_n_steps] = valid

    ext_sizes = np.zeros((n_obj, ext_len, 3), dtype=np.float32)
    ext_sizes[:, :orig_n_steps, :] = base_sizes

    step_idx = np.tile(np.arange(orig_n_steps)[None, :], (n_obj, 1))
    last_valid_idx = (valid.astype(np.int32) * step_idx).max(axis=1)

    for i in range(n_obj):
        if not ext_valid[i, current_idx]:
            continue
        last_known = int(last_valid_idx[i])
        ext_states[i, last_known + 1 :, :] = ext_states[i, last_known:last_known + 1, :]
        ext_sizes[i, orig_n_steps:, :] = ext_sizes[i, last_known:last_known + 1, :]
        ext_valid[i, last_known + 1 :] = True

    all_rollouts = np.zeros((num_rollouts, n_obj, ext_len, 4), dtype=np.float32)

    for r in range(num_rollouts):
        states = ext_states.copy()
        if current_idx + 1 < ext_len:
            states[:, current_idx + 1 :, :] = states[:, current_idx:current_idx + 1, :]

        t = current_idx
        while t < end_step:
            chunk_len = min(F, end_step - t)
            cond = build_rollout_condition_batch(
                states=states,
                valid=ext_valid,
                object_type=object_type,
                sizes=ext_sizes,
                map_cache=map_cache,
                current_t=t,
                H=HISTORY_STEPS,
                K=NEIGHBORS_K,
            )
            pred_chunk_local = sample_future_chunk(
                model,
                cond,
                diffusion_cfg=DIFFUSION_CFG,
                use_ema=use_ema,
                sample_steps=DIFFUSION_SAMPLE_STEPS,
            ).detach().cpu().numpy()
            pred_chunk_local = np.nan_to_num(pred_chunk_local, nan=0.0, posinf=0.0, neginf=0.0)

            for f in range(chunk_len):
                global_step = t + f + 1
                for i in range(n_obj):
                    prev = states[i, global_step - 1]
                    if not ext_valid[i, t]:
                        states[i, global_step] = prev
                        continue

                    anchor = states[i, t]
                    anchor_xy = anchor[:2]
                    anchor_h = float(anchor[3])

                    local_pred = pred_chunk_local[i, f]
                    desired_xy = to_world(local_pred[:2], anchor_h) + anchor_xy
                    desired_h = wrap_angle(anchor_h + math.atan2(float(local_pred[2]), float(local_pred[3])))
                    proposed = np.array([desired_xy[0], desired_xy[1], anchor[2], desired_h], dtype=np.float32)

                    if not np.all(np.isfinite(proposed)):
                        states[i, global_step] = prev
                        continue

                    next_state = apply_type_aware_clamps(prev, proposed, int(object_type[i]))
                    if not np.all(np.isfinite(next_state)):
                        states[i, global_step] = prev
                        continue

                    states[i, global_step] = next_state

            t += chunk_len

        all_rollouts[r] = states

    return all_rollouts


def predict_long_rollout_experimental(
    model: nn.Module,
    scene_features: dict,
    current_idx: int,
    horizon: int = 300,
    use_ema: bool = True,
) -> np.ndarray:
    return rollout_in_chunks(
        model=model,
        scene_features=scene_features,
        current_idx=current_idx,
        horizon=horizon,
        F=FUTURE_STEPS,
        use_ema=use_ema,
    )
