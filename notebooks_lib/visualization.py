"""Visualization helpers for report-quality notebook figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from waymo_open_dataset.utils.sim_agents import visualizations

FUTURE_STEPS = 80
TYPE_VEHICLE = 1
PAPER_FIG_DIR = Path("./figures")
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    png_path = PAPER_FIG_DIR / f"{stem}.png"
    pdf_path = PAPER_FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {png_path} and {pdf_path}")


def save_animation_gif(
    anim: animation.FuncAnimation,
    stem: str,
    fps: int = 10,
    dpi: int = 120,
) -> Path:
    """Save a Matplotlib animation as GIF under `./figures` and return its path."""
    gif_path = PAPER_FIG_DIR / f"{stem}.gif"
    try:
        writer = animation.PillowWriter(fps=fps)
    except Exception as exc:
        raise RuntimeError("GIF export requires Pillow (pip install pillow).") from exc
    anim.save(str(gif_path), writer=writer, dpi=dpi)
    print(f"Saved GIF: {gif_path}")
    return gif_path


def _future_window(current_idx: int, total_steps: int, horizon: int = FUTURE_STEPS) -> tuple[int, int]:
    start = int(current_idx + 1)
    end = int(min(total_steps, current_idx + horizon + 1))
    return start, end


def select_agents_for_plot(
    scene_features: dict,
    current_idx: int,
    max_agents: int = 12,
    prefer_vehicles: bool = True,
    type_vehicle: int = TYPE_VEHICLE,
) -> np.ndarray:
    states = scene_features["states"]
    valid = scene_features["valid"]
    object_type = scene_features["object_type"]

    n_obj, n_steps, _ = states.shape
    start, end = _future_window(current_idx, n_steps)

    scores = np.zeros((n_obj,), dtype=np.float32)
    for i in range(n_obj):
        valid_t = np.where(valid[i, start:end])[0]
        if valid_t.size < 2:
            scores[i] = -1.0
            continue
        t0 = start + int(valid_t[0])
        t1 = start + int(valid_t[-1])
        disp = states[i, t1, :2] - states[i, t0, :2]
        scores[i] = float(np.linalg.norm(disp))

    if prefer_vehicles:
        vehicle_ids = np.where(object_type == int(type_vehicle))[0]
        ranked_vehicle = vehicle_ids[np.argsort(-scores[vehicle_ids])]
        chosen = [int(i) for i in ranked_vehicle if scores[i] > 0][:max_agents]
    else:
        chosen = []

    if len(chosen) < max_agents:
        ranked_all = np.argsort(-scores)
        for i in ranked_all:
            i = int(i)
            if scores[i] <= 0:
                continue
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= max_agents:
                break

    if len(chosen) == 0:
        chosen = list(range(min(max_agents, n_obj)))

    return np.array(chosen, dtype=np.int32)


def _set_axes_focus(
    ax: plt.Axes,
    points: np.ndarray,
    pad: float = 20.0,
) -> None:
    if points.size == 0:
        return
    xmin = float(np.min(points[:, 0])) - pad
    xmax = float(np.max(points[:, 0])) + pad
    ymin = float(np.min(points[:, 1])) - pad
    ymax = float(np.max(points[:, 1])) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")


def draw_scene_overlay(
    ax: plt.Axes,
    scenario,
    scene_features: dict,
    pred_states: np.ndarray,
    current_idx: int,
    agent_ids: np.ndarray,
    title: str,
    pred_color: str = "tab:blue",
    show_gt_future: bool = True,
    show_current_bbox: bool = True,
    horizon: int = FUTURE_STEPS,
) -> None:
    states = scene_features["states"]
    valid = scene_features["valid"]
    sizes = scene_features["sizes"]
    object_type = scene_features["object_type"]

    n_steps = states.shape[1]
    start, end = _future_window(current_idx, n_steps, horizon=horizon)

    visualizations.add_map(ax, scenario)

    focus_points = []
    for i in agent_ids:
        i = int(i)
        hist_valid = valid[i, : current_idx + 1]
        if np.any(hist_valid):
            hist_xy = states[i, : current_idx + 1, :2][hist_valid]
            ax.plot(hist_xy[:, 0], hist_xy[:, 1], color="0.5", lw=1.2, alpha=0.95)
            focus_points.append(hist_xy)

        fut_valid = valid[i, start:end]
        if show_gt_future and np.any(fut_valid):
            gt_xy = states[i, start:end, :2][fut_valid]
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], "g--", lw=1.2, alpha=0.95)
            focus_points.append(gt_xy)

        pred_xy = pred_states[i, start:end, :2]
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color=pred_color, lw=1.6, alpha=0.95)
        focus_points.append(pred_xy)

        if show_current_bbox and valid[i, current_idx]:
            l = float(sizes[i, current_idx, 0])
            w = float(sizes[i, current_idx, 1])
            cur_box = visualizations.get_bbox_patch(
                float(states[i, current_idx, 0]),
                float(states[i, current_idx, 1]),
                float(states[i, current_idx, 3]),
                l,
                w,
                color_idx=int(object_type[i]) % len(visualizations.WAYMO_COLORS),
            )
            cur_box.set_alpha(0.25)
            ax.add_patch(cur_box)

    if focus_points:
        _set_axes_focus(ax, np.concatenate(focus_points, axis=0), pad=20.0)

    handles = [
        Line2D([0], [0], color="0.5", lw=1.2, label="History"),
        Line2D([0], [0], color="g", lw=1.2, ls="--", label="GT Future"),
        Line2D([0], [0], color=pred_color, lw=1.6, label="Prediction"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_main_overlay(
    scenario,
    scene_features: dict,
    pred_states: np.ndarray,
    current_idx: int,
    agent_ids: np.ndarray,
    title: str = "Qualitative Overlay",
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    draw_scene_overlay(
        ax=ax,
        scenario=scenario,
        scene_features=scene_features,
        pred_states=pred_states,
        current_idx=current_idx,
        agent_ids=agent_ids,
        title=title,
        pred_color="tab:blue",
        show_gt_future=True,
        show_current_bbox=True,
    )
    fig.tight_layout()
    return fig


def plot_keyframe_strip(
    scenario,
    scene_features: dict,
    pred_states: np.ndarray,
    current_idx: int,
    agent_ids: np.ndarray,
    keyframes: tuple[int, ...] = (10, 30, 50, 70, 90),
    title: str = "Keyframe Strip",
    type_vehicle: int = TYPE_VEHICLE,
) -> plt.Figure:
    states = scene_features["states"]
    valid = scene_features["valid"]
    sizes = scene_features["sizes"]
    object_type = scene_features["object_type"]

    n_rows = 2
    fig, axes = plt.subplots(
        n_rows,
        len(keyframes),
        figsize=(4.0 * len(keyframes), 4.2 * n_rows),
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif axes.ndim == 1:
        axes = axes.reshape(n_rows, 1)

    for col, k in enumerate(keyframes):
        ax = axes[0, col]
        t = int(min(max(0, k), states.shape[1] - 1))
        visualizations.add_map(ax, scenario)

        focus_points = []
        for i in agent_ids:
            i = int(i)
            hist_end = min(t, current_idx)
            if hist_end >= 0:
                hist_valid = valid[i, : hist_end + 1]
                if np.any(hist_valid):
                    hist_xy = states[i, : hist_end + 1, :2][hist_valid]
                    ax.plot(hist_xy[:, 0], hist_xy[:, 1], color="0.6", lw=1.0)
                    focus_points.append(hist_xy)

            start = current_idx + 1
            if t >= start:
                pred_xy = pred_states[i, start : t + 1, :2]
                ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="tab:blue", lw=1.4)
                focus_points.append(pred_xy)

            if t < states.shape[1] and valid[i, t]:
                l = float(sizes[i, min(t, sizes.shape[1]-1), 0])
                w = float(sizes[i, min(t, sizes.shape[1]-1), 1])
                box = visualizations.get_bbox_patch(
                    float(pred_states[i, t, 0]),
                    float(pred_states[i, t, 1]),
                    float(pred_states[i, t, 3]),
                    l,
                    w,
                    color_idx=int(object_type[i]) % len(visualizations.WAYMO_COLORS),
                )
                box.set_alpha(0.28)
                ax.add_patch(box)

        if focus_points:
            _set_axes_focus(ax, np.concatenate(focus_points, axis=0), pad=20.0)

        ax.set_title(f"t={t}")
        ax.set_xlabel("x (m)")
        if col == 0:
            ax.set_ylabel("y (m)")

        ax_box = axes[1, col]
        visualizations.add_map(ax_box, scenario)

        vehicle_agents = [int(i) for i in agent_ids if int(object_type[int(i)]) == int(type_vehicle)]
        if not vehicle_agents:
            vehicle_agents = [int(i) for i in agent_ids]

        box_focus = []
        for i in vehicle_agents:
            if t >= states.shape[1] or not valid[i, t]:
                continue
            l = float(sizes[i, min(t, sizes.shape[1] - 1), 0])
            w = float(sizes[i, min(t, sizes.shape[1] - 1), 1])
            box = visualizations.get_bbox_patch(
                float(pred_states[i, t, 0]),
                float(pred_states[i, t, 1]),
                float(pred_states[i, t, 3]),
                l,
                w,
                color_idx=int(object_type[i]) % len(visualizations.WAYMO_COLORS),
            )
            box.set_alpha(0.55)
            ax_box.add_patch(box)
            box_focus.append(np.array([[pred_states[i, t, 0], pred_states[i, t, 1]]], dtype=np.float32))

        if box_focus:
            _set_axes_focus(ax_box, np.concatenate(box_focus, axis=0), pad=20.0)

        ax_box.set_title(f"t={t} vehicle only")
        ax_box.set_xlabel("x (m)")
        if col == 0:
            ax_box.set_ylabel("y (m)")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def compute_horizon_error_curve(
    pred_states: np.ndarray,
    gt_states: np.ndarray,
    gt_valid: np.ndarray,
    current_idx: int,
    horizon: int = FUTURE_STEPS,
) -> np.ndarray:
    start, end = _future_window(current_idx, gt_states.shape[1], horizon=horizon)
    err = []
    for t in range(start, end):
        mask = gt_valid[:, t]
        if np.any(mask):
            d = np.linalg.norm(pred_states[:, t, :2] - gt_states[:, t, :2], axis=1)
            err.append(float(np.mean(d[mask])))
        else:
            err.append(np.nan)
    return np.array(err, dtype=np.float32)


def plot_error_horizon_curves(
    curves: dict,
    title: str = "Error Over Horizon",
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for name, arr in curves.items():
        arr = np.asarray(arr, dtype=np.float32)
        x = np.arange(1, len(arr) + 1)
        ax.plot(x, arr, lw=1.8, label=name)
    ax.set_xlabel("Future step (1..80)")
    ax.set_ylabel("Mean displacement error (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_safety_bar(ablation: dict, title: str = "Safety Metrics by Mode") -> plt.Figure:
    modes = list(ablation.keys())
    offroad = [float(ablation[m]["offroad_violations"]) for m in modes]
    collisions = [float(ablation[m]["collision_pairs"]) for m in modes]
    slip = [float(ablation[m]["mean_vehicle_slip_ratio"]) for m in modes]

    x = np.arange(len(modes))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.bar(x - width, offroad, width=width, label="Off-road")
    ax.bar(x, collisions, width=width, label="Collisions")
    ax.bar(x + width, slip, width=width, label="Slip ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_runtime_pareto(summary_df, title: str = "Runtime vs Quality Pareto") -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if summary_df is None or len(summary_df) == 0:
        ax.text(0.5, 0.5, "No summary_df available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x = summary_df["runtime_s"].values
    y = summary_df["model_ade80"].values
    c = summary_df["final_score"].values

    sc = ax.scatter(x, y, c=c, cmap="viridis", s=35)
    ax.set_xlabel("Runtime per run (s)")
    ax.set_ylabel("ADE80 (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    if "config_id" in summary_df.columns and len(summary_df) > 0:
        best = summary_df.iloc[0]
        ax.scatter([best["runtime_s"]], [best["model_ade80"]], marker="*", s=180, color="red", label="Selected")
        ax.legend()

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("final_score")
    fig.tight_layout()
    return fig


def plot_case_gallery(
    scenario,
    scene_features: dict,
    current_idx: int,
    agent_ids: np.ndarray,
    cases: list,
    title: str = "Case Gallery",
) -> plt.Figure:
    n = len(cases)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5 * n_cols, 7.0 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, case in zip(axes, cases):
        name, pred_states, pred_color = case
        draw_scene_overlay(
            ax=ax,
            scenario=scenario,
            scene_features=scene_features,
            pred_states=pred_states,
            current_idx=current_idx,
            agent_ids=agent_ids,
            title=name,
            pred_color=pred_color,
            show_gt_future=True,
            show_current_bbox=False,
        )

    for i in range(len(cases), len(axes)):
        axes[i].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig

def plot_gt_vs_prediction_animation(
    scenario,
    scene_features: dict,
    my_simulated_states: np.ndarray,
    sample_index: int = 0,
    current_idx: int = 10,
    total_steps: int = 91,
    interval: int = 100,
    save_gif: bool = True,
) -> animation.FuncAnimation:
    type_names = {1: "Veh", 2: "Ped", 3: "Cyc"}

    agent_data = []
    for ai, track in enumerate(scenario.tracks):
        states = track.states
        if len(states) < current_idx + 1:
            continue
        curr = states[current_idx]
        if not curr.valid:
            continue
        xs = [s.center_x if s.valid else np.nan for s in states[:total_steps]]
        ys = [s.center_y if s.valid else np.nan for s in states[:total_steps]]
        speed = np.sqrt(curr.velocity_x**2 + curr.velocity_y**2)
        label = f"{ai}:{type_names.get(track.object_type,'?')} {speed:.1f}m/s"
        agent_data.append((ai, xs, ys, label))

    all_xs = [x for _, xs, _, _ in agent_data for x in xs if not np.isnan(x)]
    all_ys = [y for _, _, ys, _ in agent_data for y in ys if not np.isnan(y)]
    pad = 40

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    for ax, title in [(ax1,"Ground Truth"), (ax2, "Model Prediction")]:
        ax.set_facecolor("white")
        visualizations.add_map(ax, scenario)
        ax.set_xlim(min(all_xs)-pad, max(all_xs)+pad)
        ax.set_ylim(min(all_ys)-pad, max(all_ys)+pad)
        ax.set_aspect("equal")
        ax.tick_params(colors="black")
        ax.set_title(title, color="black", fontsize=14, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    fig.patch.set_facecolor("white")
    fig.suptitle(f"Scenario {scenario.scenario_id}", color = "black", fontsize = 16, fontweight = "bold", y = 0.98)

    gt_trails = [ax1.plot([], [], "-", color = "red", lw = 1.5, alpha = 0.6)[0] for _ in agent_data]
    gt_dots   = [ax1.plot([], [], "o", color = "dodgerblue", ms = 9, zorder = 6)[0] for _ in agent_data]
    gt_labels = [ax1.text(0, 0, "", fontsize = 7, color = "black", zorder = 7) for _ in agent_data]

    md_trails = [ax2.plot([], [], "-", color = "red", lw = 1.5, alpha = 0.6)[0] for _ in agent_data]
    md_dots   = [ax2.plot([], [], "o", color = "dodgerblue", ms=9, zorder = 6)[0] for _ in agent_data]
    md_labels = [ax2.text(0, 0, "", fontsize = 7, color = "black", zorder = 7) for _ in agent_data]

    time_text = ax1.text(0.02, 0.97, "", transform = ax1.transAxes, color = "black", fontsize = 11, va = "top", fontweight = "bold")

    def update(frame):
        time_text.set_text(f"t = {frame - current_idx:+d}  ({(frame - current_idx)*0.1:+.1f}s)")
        for idx, (ai, xs, ys, label) in enumerate(agent_data):
            trail_x = [x for x in xs[:frame+1] if not np.isnan(x)]
            trail_y = [y for y in ys[:frame+1] if not np.isnan(y)]
            gt_trails[idx].set_data(trail_x, trail_y)
            if not np.isnan(xs[frame]) and not np.isnan(ys[frame]):
                gt_dots[idx].set_data([xs[frame]], [ys[frame]])
                gt_labels[idx].set_position((xs[frame] + 2, ys[frame] + 2))
                gt_labels[idx].set_text(label)
            else:
                gt_dots[idx].set_data([], [])
                gt_labels[idx].set_text("")

            if ai < my_simulated_states.shape[1]:
                mx = my_simulated_states[sample_index, ai, frame, 0]
                my = my_simulated_states[sample_index, ai, frame, 1]
                md_trails[idx].set_data(
                    [my_simulated_states[sample_index, ai, f, 0] for f in range(frame+1)],
                    [my_simulated_states[sample_index, ai, f, 1] for f in range(frame+1)],
                )
                md_dots[idx].set_data([mx], [my])
                md_labels[idx].set_position((mx + 2, my + 2))
                md_labels[idx].set_text(label)

        return gt_trails + gt_dots + gt_labels + md_trails + md_dots + md_labels + [time_text]

    ani = animation.FuncAnimation(
        fig, update,
        frames = range(0, total_steps),
        interval = interval,
        blit = False,
    )
    plt.tight_layout()

    if save_gif:
        gif_path = save_animation_gif(ani, stem = f"comparison_{scenario.scenario_id}", fps = 10, dpi = 100)
        print(f"GIF saved: {gif_path}")

    plt.close(fig)
    return ani

