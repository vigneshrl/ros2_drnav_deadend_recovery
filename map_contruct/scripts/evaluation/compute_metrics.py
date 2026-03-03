#!/usr/bin/env python3
"""
Compute DR.Nav Table II metrics from ROS2 bag files.

Metrics
-------
  dist (m)      Total cumulative Euclidean displacement
                  = Σ_t ||p(t) - p(t-1)||  from /odom at 60 Hz
  speed (m/s)   Mean forward speed = dist / duration
  efficiency    Straight-line(start→end) / actual path length
  NPR           Negative Progress Ratio — fraction of path that reduces
                  progress toward the current goal (backtracking / lateral)
                  Lower is better.
  PAD (m)       Pre-emptive Avoidance Distance — distance from the dead-end
                  mouth when the planner first turns away.
                  Higher means more proactive (DRAM > DWA expected).

Dead-end episodes are defined as contiguous segments where
/dead_end_detection/is_dead_end stays True for ≥ 2 consecutive seconds
(same filter as in the paper).

Usage
-----
Single bag:
    python3 compute_metrics.py /path/to/bag_dir --method DR_Nav

Multiple bags (averaged over runs):
    python3 compute_metrics.py run1/ run2/ run3/ run4/ run5/ --method DR_Nav

Recording bags (run before each navigation trial):
    ros2 bag record \\
        /odom \\
        /move_base_simple/goal \\
        /dead_end_detection/is_dead_end \\
        /cmd_vel \\
        -o bags/dram_run1
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    sys.exit(
        "ERROR: rosbag2_py / rclpy not found.\n"
        "Source your ROS2 workspace:  source /opt/ros/humble/setup.bash"
    )


# ── helpers ────────────────────────────────────────────────────────────────────

def _norm(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else (0.0, 0.0)

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _angle_diff(h1, h2):
    d = abs(h1 - h2) % (2 * math.pi)
    return d if d <= math.pi else 2 * math.pi - d


# ── bag reader ─────────────────────────────────────────────────────────────────

WANTED_TOPICS = {
    '/odom',
    '/odom_lidar',
    '/move_base_simple/goal',
    '/dead_end_detection/is_dead_end',
    '/cmd_vel',
}


def _open_reader(bag_path):
    """Try sqlite3 then mcap storage IDs."""
    reader = rosbag2_py.SequentialReader()
    conv = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    for sid in ['', 'sqlite3', 'mcap']:
        try:
            reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=sid), conv)
            return reader
        except Exception:
            pass
    raise RuntimeError(f"Cannot open bag: {bag_path}")


def read_bag(bag_path):
    """Return dict: topic → sorted list of (timestamp_ns, deserialized_msg)."""
    reader = _open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    messages = defaultdict(list)

    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic not in WANTED_TOPICS:
            continue
        try:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            messages[topic].append((ts, msg))
        except Exception:
            pass

    return messages


# ── data extraction ────────────────────────────────────────────────────────────

def _poses(messages):
    """Sorted list of (ts_ns, x, y) from /odom_lidar or /odom."""
    topic = '/odom_lidar' if '/odom_lidar' in messages else '/odom'
    out = []
    for ts, msg in messages.get(topic, []):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        out.append((ts, x, y))
    return sorted(out)


def _goals(messages):
    """Sorted list of (ts_ns, gx, gy)."""
    out = []
    for ts, msg in messages.get('/move_base_simple/goal', []):
        out.append((ts, msg.pose.position.x, msg.pose.position.y))
    return sorted(out)


def _dead_end_flags(messages):
    """Sorted list of (ts_ns, bool)."""
    out = []
    for ts, msg in messages.get('/dead_end_detection/is_dead_end', []):
        out.append((ts, bool(msg.data)))
    return sorted(out)


def _goal_at(goals, ts_ns):
    """Most recent goal published at or before ts_ns. Returns (gx, gy) or None."""
    current = None
    for gts, gx, gy in goals:
        if gts <= ts_ns:
            current = (gx, gy)
        else:
            break
    return current


# ── metric computation ─────────────────────────────────────────────────────────

def compute_metrics(bag_path, de_min_duration_s=2.0, pad_lookahead_s=10.0,
                    turn_threshold_deg=60.0):
    """
    Returns dict with keys:
        dist, speed, efficiency, npr, pad, duration, n_dead_ends
    """
    messages = read_bag(bag_path)
    poses    = _poses(messages)
    goals    = _goals(messages)
    de_flags = _dead_end_flags(messages)

    if len(poses) < 2:
        print(f"  WARNING: only {len(poses)} pose messages — skipping.")
        return None

    # ── distance / NPR ────────────────────────────────────────────────────────
    total_dist     = 0.0
    backtrack_dist = 0.0
    prev_pos = (poses[0][1], poses[0][2])

    for ts, x, y in poses[1:]:
        dp   = (x - prev_pos[0], y - prev_pos[1])
        step = math.hypot(*dp)
        if step < 1e-6:
            prev_pos = (x, y)
            continue
        total_dist += step

        goal = _goal_at(goals, ts)
        if goal is not None:
            goal_dir = _norm((goal[0] - prev_pos[0], goal[1] - prev_pos[1]))
            if _dot(_norm(dp), goal_dir) < 0:   # moving away from goal
                backtrack_dist += step

        prev_pos = (x, y)

    duration_s  = (poses[-1][0] - poses[0][0]) * 1e-9
    start_pos   = (poses[0][1],  poses[0][2])
    end_pos     = (poses[-1][1], poses[-1][2])
    straight    = _dist(start_pos, end_pos)

    avg_speed   = total_dist / duration_s   if duration_s  > 0    else 0.0
    efficiency  = straight   / total_dist   if total_dist  > 1e-3 else 0.0
    npr         = backtrack_dist / total_dist if total_dist > 1e-3 else 0.0

    # ── PAD ───────────────────────────────────────────────────────────────────
    # Build numpy arrays for fast interpolation of robot position
    pose_ts = np.array([p[0] for p in poses], dtype=np.float64)
    pose_x  = np.array([p[1] for p in poses], dtype=np.float64)
    pose_y  = np.array([p[2] for p in poses], dtype=np.float64)

    def robot_pos_at(ts_ns):
        idx = int(np.searchsorted(pose_ts, ts_ns))
        idx = max(0, min(idx, len(poses) - 1))
        return (float(pose_x[idx]), float(pose_y[idx]))

    # Detect dead-end episodes
    episodes = []
    ep_start_ts  = None
    ep_start_pos = None
    prev_de = False

    for ts, is_de in de_flags:
        if is_de and not prev_de:
            ep_start_ts  = ts
            ep_start_pos = robot_pos_at(ts)
        elif not is_de and prev_de and ep_start_ts is not None:
            if (ts - ep_start_ts) * 1e-9 >= de_min_duration_s:
                episodes.append({'mouth_pos': ep_start_pos,
                                  'mouth_ts':  ep_start_ts})
            ep_start_ts = None
        prev_de = is_de

    # Handle episode still open at end of bag
    if prev_de and ep_start_ts is not None:
        if (poses[-1][0] - ep_start_ts) * 1e-9 >= de_min_duration_s:
            episodes.append({'mouth_pos': ep_start_pos,
                              'mouth_ts':  ep_start_ts})

    pad_values = []
    turn_thresh_rad = math.radians(turn_threshold_deg)

    for ep in episodes:
        mouth_pos = ep['mouth_pos']
        mouth_ts  = ep['mouth_ts']
        search_start = mouth_ts - int(pad_lookahead_s * 1e9)

        # Goals published in the lookahead window before dead-end detection
        window_goals = [(gts, gx, gy) for gts, gx, gy in goals
                        if search_start <= gts <= mouth_ts]

        if len(window_goals) < 2:
            # No goal changes recorded — assume reactive (PAD = 0)
            pad_values.append(0.0)
            continue

        # Find the first large goal direction change in this window
        turn_away_pos = None
        prev_heading  = None

        for gts, gx, gy in window_goals:
            rpos = robot_pos_at(gts)
            hdg  = math.atan2(gy - rpos[1], gx - rpos[0])
            if prev_heading is not None:
                if _angle_diff(hdg, prev_heading) > turn_thresh_rad:
                    turn_away_pos = rpos
                    break
            prev_heading = hdg

        if turn_away_pos is not None:
            pad_values.append(_dist(turn_away_pos, mouth_pos))
        else:
            # No significant turn before dead-end (reactive planner)
            pad_values.append(0.0)

    pad = float(np.mean(pad_values)) if pad_values else float('nan')

    return {
        'dist':        total_dist,
        'speed':       avg_speed,
        'efficiency':  efficiency,
        'npr':         npr,
        'pad':         pad,
        'duration':    duration_s,
        'n_dead_ends': len(episodes),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute DR.Nav Table II metrics from ROS2 bags.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('bags', nargs='+',
                        help='Bag directory paths (one or more runs)')
    parser.add_argument('--method', default='unknown',
                        help='Method name label (e.g. DR_Nav, DWA, MPPI)')
    parser.add_argument('--de-min-duration', type=float, default=2.0,
                        help='Min dead-end episode duration in seconds (default 2.0)')
    parser.add_argument('--pad-lookahead', type=float, default=10.0,
                        help='Seconds before dead-end detection to search for turn (default 10)')
    parser.add_argument('--turn-threshold', type=float, default=60.0,
                        help='Goal heading change in degrees to count as "turning away" (default 60)')
    args = parser.parse_args()

    results = []
    for bag_path in args.bags:
        print(f"\nProcessing: {bag_path}")
        try:
            r = compute_metrics(
                bag_path,
                de_min_duration_s=args.de_min_duration,
                pad_lookahead_s=args.pad_lookahead,
                turn_threshold_deg=args.turn_threshold,
            )
            if r is not None:
                results.append(r)
                pad_str = f"{r['pad']:.2f}" if not math.isnan(r['pad']) else "n/a"
                print(f"  dist={r['dist']:.1f}m  speed={r['speed']:.3f}m/s  "
                      f"eff={r['efficiency']:.3f}  PAD={pad_str}m  NPR={r['npr']:.2f}  "
                      f"dead_ends={r['n_dead_ends']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("\nNo valid results extracted.")
        return

    # Aggregate
    def _agg(key):
        vals = [r[key] for r in results if not math.isnan(r[key])]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), 0.0)

    d_m, d_s   = _agg('dist')
    sp_m, sp_s = _agg('speed')
    ef_m, ef_s = _agg('efficiency')
    np_m, np_s = _agg('npr')
    pa_m, pa_s = _agg('pad')

    print(f"\n{'='*62}")
    print(f"  Method : {args.method}   ({len(results)} run(s))")
    print(f"{'='*62}")
    print(f"  Distance   (↓) : {d_m:7.1f} ± {d_s:.1f} m")
    print(f"  Speed    (↑)   : {sp_m:.3f} ± {sp_s:.3f} m/s")
    print(f"  Efficiency (↑) : {ef_m:.3f} ± {ef_s:.3f}")
    print(f"  PAD      (↑)   : {pa_m:.2f} ± {pa_s:.2f} m  (higher = more proactive)")
    print(f"  NPR      (↓)   : {np_m:.2f} ± {np_s:.2f}    (lower = less backtracking)")
    print(f"{'='*62}")
    print(f"\n  Table II row:")
    print(f"  {args.method:<22s}  {d_m:6.1f}  {sp_m:.3f}  {ef_m:.3f}  {pa_m:.1f}  {np_m:.2f}")
    print()


if __name__ == '__main__':
    main()
