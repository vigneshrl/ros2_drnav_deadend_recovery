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
    '/dead_end_detection/path_status',
    '/cmd_vel',
    '/tf',
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
    """Sorted list of (ts_ns, x, y) in the MAP frame.

    Goals are published in the map frame, so NPR and PAD are only meaningful
    if robot poses are also in the map frame.

    Strategy:
      1. Reconstruct map→base_link by composing map→odom and odom→base_link
         from the recorded /tf topic.
      2. Fall back to /odom_lidar (odom frame) if TF data is absent.
    """
    # ── attempt TF reconstruction ────────────────────────────────────────────
    tf_msgs = messages.get('/tf', [])
    # Build lookup: (parent, child) → sorted list of (ts_ns, tx, ty, qz, qw)
    tf_tree = defaultdict(list)
    for ts, msg in tf_msgs:
        for t in msg.transforms:
            key = (t.header.frame_id, t.child_frame_id)
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w
            tf_tree[key].append((ts, tx, ty, qz, qw))
    for key in tf_tree:
        tf_tree[key].sort()

    def _lookup_tf(key, ts_ns):
        """Nearest-in-time TF lookup. Returns (tx, ty, qz, qw) or None."""
        entries = tf_tree.get(key)
        if not entries:
            return None
        idx = int(np.searchsorted([e[0] for e in entries], ts_ns))
        idx = max(0, min(idx, len(entries) - 1))
        _, tx, ty, qz, qw = entries[idx]
        return tx, ty, qz, qw

    def _compose(t1, t2):
        """Compose two 2-D transforms (tx,ty,qz,qw) : T_result = T1 ∘ T2."""
        tx1, ty1, qz1, qw1 = t1
        tx2, ty2, qz2, qw2 = t2
        # Rotate T2 translation by T1 heading
        s1 = 2 * qw1 * qz1       # sin(yaw1)
        c1 = 1 - 2 * qz1 * qz1   # cos(yaw1)
        tx = tx1 + c1 * tx2 - s1 * ty2
        ty = ty1 + s1 * tx2 + c1 * ty2
        # Quaternion multiply (2-D: only z,w components)
        qz = qw1 * qz2 + qz1 * qw2
        qw = qw1 * qw2 - qz1 * qz2
        return tx, ty, qz, qw

    have_map_odom   = bool(tf_tree.get(('map',  'odom')))
    have_odom_base  = bool(tf_tree.get(('odom', 'base_link')))
    use_tf = have_map_odom and have_odom_base

    if use_tf:
        # Build poses in map frame via composed TF
        odom_topic = '/odom_lidar' if '/odom_lidar' in messages else '/odom'
        out = []
        for ts, msg in messages.get(odom_topic, []):
            # odom→base_link from the recorded /tf
            t_ob = _lookup_tf(('odom', 'base_link'), ts)
            t_mo = _lookup_tf(('map',  'odom'),       ts)
            if t_ob is None or t_mo is None:
                continue
            tx, ty, _, _ = _compose(t_mo, t_ob)
            out.append((ts, tx, ty))
        if out:
            return sorted(out)
        # fall through if compose produced nothing

    # ── fallback: use raw odom position ─────────────────────────────────────
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


def _path_status(messages):
    """Sorted list of (ts_ns, F, L, R) from /dead_end_detection/path_status."""
    out = []
    for ts, msg in messages.get('/dead_end_detection/path_status', []):
        if len(msg.data) >= 3:
            out.append((ts, float(msg.data[0]), float(msg.data[1]), float(msg.data[2])))
    return sorted(out)


def compute_detection_latency(messages, block_threshold=0.45, de_min_duration_s=2.0):
    """
    Detection latency: seconds from when path scores first ALL drop below
    block_threshold to when is_dead_end first becomes True in that episode.

    Also computes false-positive rate: fraction of 1-second open-space windows
    (all path scores above threshold) where is_dead_end=True fires.

    Returns (latencies_list, fp_rate, n_fp_windows, n_open_windows)
    """
    ps      = _path_status(messages)
    de_flags = _dead_end_flags(messages)

    if not ps or not de_flags:
        return [], float('nan'), 0, 0

    # Build is_dead_end lookup
    de_ts_arr  = np.array([t for t, _ in de_flags])
    de_val_arr = np.array([v for _, v in de_flags], dtype=bool)

    def is_de_at(ts_ns):
        idx = int(np.searchsorted(de_ts_arr, ts_ns))
        idx = max(0, min(idx, len(de_ts_arr) - 1))
        return bool(de_val_arr[idx])

    # ── Detection latency ────────────────────────────────────────────────────
    # Find contiguous blocks where all 3 scores < block_threshold
    latencies = []
    in_block  = False
    block_start_ts = None

    for ts, F, L, R in ps:
        all_blocked = F < block_threshold and L < block_threshold and R < block_threshold
        if all_blocked and not in_block:
            in_block       = True
            block_start_ts = ts
        elif not all_blocked and in_block:
            in_block = False
            block_start_ts = None

        if in_block and is_de_at(ts):
            # First callback where both model says blocked AND is_dead_end=True
            latency_s = (ts - block_start_ts) * 1e-9
            if 0.0 < latency_s < 10.0:   # sanity filter
                latencies.append(latency_s)
            in_block = False
            block_start_ts = None

    # ── False-positive rate ──────────────────────────────────────────────────
    # 1-second sliding windows where all scores > block_threshold (open space)
    # Count how many such windows have is_dead_end=True at any point inside
    window_ns    = int(1.0 * 1e9)
    n_open       = 0
    n_fp         = 0
    ps_arr       = np.array([(t, F, L, R) for t, F, L, R in ps])

    i = 0
    while i < len(ps):
        ts, F, L, R = ps[i]
        if F > block_threshold and L > block_threshold and R > block_threshold:
            # Open-space window: check the next 1 second
            window_end = ts + window_ns
            j = i
            all_open_in_window = True
            fp_in_window       = False
            while j < len(ps) and ps[j][0] <= window_end:
                wF, wL, wR = ps[j][1], ps[j][2], ps[j][3]
                if wF < block_threshold or wL < block_threshold or wR < block_threshold:
                    all_open_in_window = False
                    break
                if is_de_at(ps[j][0]):
                    fp_in_window = True
                j += 1
            if all_open_in_window and j > i:
                n_open += 1
                if fp_in_window:
                    n_fp += 1
            i = j if j > i else i + 1
        else:
            i += 1

    fp_rate = n_fp / n_open if n_open > 0 else float('nan')
    return latencies, fp_rate, n_fp, n_open


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

    # Detect dead-end episodes and measure time-to-exit / recovery success
    # A recovery is "successful" when:
    #   - is_dead_end returns to False (robot exited)
    #   - robot has moved > exit_dist_m from the mouth position
    exit_dist_m = 1.0   # metres robot must move away from mouth to count as exited

    episodes        = []
    ep_start_ts     = None
    ep_start_pos    = None
    ep_end_ts       = None
    prev_de         = False
    n_recovered     = 0
    exit_times      = []   # seconds from mouth to exit for successful recoveries

    for ts, is_de in de_flags:
        if is_de and not prev_de:
            ep_start_ts  = ts
            ep_start_pos = robot_pos_at(ts)
            ep_end_ts    = None
        elif not is_de and prev_de and ep_start_ts is not None:
            ep_end_ts = ts
            duration_ep = (ep_end_ts - ep_start_ts) * 1e-9
            if duration_ep >= de_min_duration_s:
                exit_pos = robot_pos_at(ep_end_ts)
                exited   = _dist(exit_pos, ep_start_pos) >= exit_dist_m
                episodes.append({'mouth_pos':  ep_start_pos,
                                  'mouth_ts':   ep_start_ts,
                                  'end_ts':     ep_end_ts,
                                  'exited':     exited,
                                  'duration':   duration_ep})
                if exited:
                    n_recovered += 1
                    exit_times.append(duration_ep)
            ep_start_ts = None
        prev_de = is_de

    # Handle episode still open at end of bag (robot never exited)
    if prev_de and ep_start_ts is not None:
        duration_ep = (poses[-1][0] - ep_start_ts) * 1e-9
        if duration_ep >= de_min_duration_s:
            episodes.append({'mouth_pos': ep_start_pos,
                              'mouth_ts':  ep_start_ts,
                              'end_ts':    None,
                              'exited':    False,
                              'duration':  duration_ep})

    n_dead_ends   = len(episodes)
    recovery_rate = n_recovered / n_dead_ends if n_dead_ends > 0 else float('nan')
    mean_exit_t   = float(np.mean(exit_times)) if exit_times else float('nan')

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

        # Reference heading: direction the robot was heading just before the
        # dead-end mouth (last goal in the window = "into dead end" direction).
        # We compare every earlier goal against this reference rather than
        # against the previous step.  This captures gradual EDE-guided avoidance
        # where each individual step change is small (< 60°) but the cumulative
        # deviation from the dead-end heading is large.
        last_gts, last_gx, last_gy = window_goals[-1]
        last_rpos   = robot_pos_at(last_gts)
        dead_end_hdg = math.atan2(last_gy - last_rpos[1], last_gx - last_rpos[0])

        turn_away_pos = None

        for gts, gx, gy in window_goals[:-1]:   # search all goals except the last
            rpos = robot_pos_at(gts)
            hdg  = math.atan2(gy - rpos[1], gx - rpos[0])
            if _angle_diff(hdg, dead_end_hdg) > turn_thresh_rad:
                turn_away_pos = rpos
                break   # earliest point with sufficient deviation from dead-end

        if turn_away_pos is not None:
            pad_values.append(_dist(turn_away_pos, mouth_pos))
        else:
            # No significant deviation from dead-end heading — reactive planner
            pad_values.append(0.0)

    pad = float(np.mean(pad_values)) if pad_values else float('nan')

    return {
        'dist':          total_dist,
        'speed':         avg_speed,
        'efficiency':    efficiency,
        'npr':           npr,
        'pad':           pad,
        'duration':      duration_s,
        'n_dead_ends':   n_dead_ends,
        'recovery_rate': recovery_rate,   # fraction of episodes robot exited
        'mean_exit_t':   mean_exit_t,     # mean seconds from detection to exit
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

    all_latencies = []
    all_fp_rates  = []
    all_open_wins = 0

    results = []
    for bag_path in args.bags:
        print(f"\nProcessing: {bag_path}")
        try:
            # Detection latency + false-positive rate
            messages_raw = read_bag(bag_path)
            lats, fpr, n_fp, n_open = compute_detection_latency(messages_raw)
            if lats:
                all_latencies.extend(lats)
                print(f"  detection latency per episode: {[f'{l:.2f}s' for l in lats]}")
            if not math.isnan(fpr):
                all_fp_rates.append(fpr)
                all_open_wins += n_open
                print(f"  false-positive rate: {fpr*100:.1f}%  ({n_fp}/{n_open} open windows)")

            r = compute_metrics(
                bag_path,
                de_min_duration_s=args.de_min_duration,
                pad_lookahead_s=args.pad_lookahead,
                turn_threshold_deg=args.turn_threshold,
            )
            if r is not None:
                results.append(r)
                pad_str = f"{r['pad']:.2f}"   if not math.isnan(r['pad'])           else "n/a"
                rr_str  = f"{r['recovery_rate']*100:.0f}%" if not math.isnan(r['recovery_rate']) else "n/a"
                te_str  = f"{r['mean_exit_t']:.1f}s"      if not math.isnan(r['mean_exit_t'])    else "n/a"
                print(f"  dist={r['dist']:.1f}m  speed={r['speed']:.3f}m/s  "
                      f"eff={r['efficiency']:.3f}  PAD={pad_str}m  NPR={r['npr']:.2f}  "
                      f"dead_ends={r['n_dead_ends']}  recovery={rr_str}  exit_t={te_str}")
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
    rr_m, rr_s = _agg('recovery_rate')
    te_m, te_s = _agg('mean_exit_t')
    total_de   = sum(r['n_dead_ends'] for r in results)

    print(f"\n{'='*62}")
    print(f"  Method : {args.method}   ({len(results)} run(s))")
    print(f"{'='*62}")
    print(f"  Distance       (↓) : {d_m:7.1f} ± {d_s:.1f} m")
    print(f"  Speed          (↑) : {sp_m:.3f} ± {sp_s:.3f} m/s")
    print(f"  Efficiency     (↑) : {ef_m:.3f} ± {ef_s:.3f}")
    print(f"  PAD            (↑) : {pa_m:.2f} ± {pa_s:.2f} m  (higher = more proactive)")
    print(f"  NPR            (↓) : {np_m:.2f} ± {np_s:.2f}    (lower = less backtracking)")
    print(f"  Dead-end episodes  : {total_de} total across {len(results)} run(s)")
    if not math.isnan(rr_m):
        print(f"  Recovery rate  (↑) : {rr_m*100:.0f}% ± {rr_s*100:.0f}%  (episodes exited)")
        print(f"  Mean exit time (↓) : {te_m:.1f} ± {te_s:.1f} s")
    else:
        print(f"  Recovery rate      : n/a (no dead-end episodes or no detector)")
    print(f"{'='*62}")
    print(f"\n  Table II row:")
    print(f"  {args.method:<22s}  {d_m:6.1f}  {sp_m:.3f}  {ef_m:.3f}  {pa_m:.1f}  {np_m:.2f}")

    # Section G values
    if all_latencies:
        lat_mean = float(np.mean(all_latencies))
        lat_std  = float(np.std(all_latencies))
        print(f"\n  [Sec G] Detection latency : {lat_mean:.2f} ± {lat_std:.2f} s  "
              f"(over {len(all_latencies)} episode(s))")
    if all_fp_rates:
        fp_mean = float(np.mean(all_fp_rates)) * 100
        print(f"  [Sec G] False-positive rate: {fp_mean:.1f}%  "
              f"({all_open_wins} open-space windows total)")
    print()


if __name__ == '__main__':
    main()
