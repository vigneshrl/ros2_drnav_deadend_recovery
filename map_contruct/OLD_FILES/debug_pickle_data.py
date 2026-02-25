#!/usr/bin/env python3

import pickle
import os

def debug_pickle_data(pickle_path):
    """Debug why all values are zero"""
    
    print(f"🔍 DEBUGGING PICKLE DATA: {pickle_path}")
    print("=" * 60)
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if this is the main action log with more detailed info
    main_action_file = pickle_path.replace('camera_actions.pkl', 'action_log.pkl')
    
    if os.path.exists(main_action_file):
        print(f"📂 Found main action log: {main_action_file}")
        with open(main_action_file, 'rb') as f:
            main_data = pickle.load(f)
        
        print("\n📊 MAIN ACTION LOG ANALYSIS:")
        print("-" * 40)
        
        if 'actions' in main_data:
            regular_actions = main_data['actions']
            print(f"🎬 Total regular actions: {len(regular_actions)}")
            
            if len(regular_actions) > 0:
                # Check if regular actions also have zero velocities
                velocities = [action.get('v', 0) for action in regular_actions]
                omegas = [action.get('omega', 0) for action in regular_actions]
                
                print(f"📈 Regular action velocity range: {min(velocities):.3f} to {max(velocities):.3f}")
                print(f"📈 Regular action omega range: {min(omegas):.3f} to {max(omegas):.3f}")
                
                # Show some sample regular actions
                print(f"\n📋 SAMPLE REGULAR ACTIONS:")
                for i, action in enumerate(regular_actions[:5]):
                    print(f"  {i+1}. v={action.get('v', 'N/A')}, ω={action.get('omega', 'N/A')}, "
                          f"pos=({action.get('x', 'N/A')}, {action.get('y', 'N/A')})")
        
        if 'session_info' in main_data:
            session = main_data['session_info']
            print(f"\n📝 SESSION INFO:")
            print(f"   Duration: {session.get('end_time', 0) - session.get('start_time', 0):.2f} seconds")
            print(f"   Total actions: {session.get('total_actions', 0)}")
            print(f"   Camera frames: {session.get('total_camera_frames', 0)}")
    
    # Analyze camera actions
    if 'camera_timestamp_to_action' in data:
        camera_actions = data['camera_timestamp_to_action']
        print(f"\n📸 CAMERA ACTION ANALYSIS:")
        print("-" * 40)
        print(f"🎬 Total camera frames: {len(camera_actions)}")
        
        # Check for any non-zero values
        all_v = [action['v'] for action in camera_actions.values()]
        all_omega = [action['omega'] for action in camera_actions.values()]
        all_x = [action['robot_pose'][0] for action in camera_actions.values()]
        all_y = [action['robot_pose'][1] for action in camera_actions.values()]
        all_theta = [action['robot_pose'][2] for action in camera_actions.values()]
        
        print(f"📊 STATISTICS:")
        print(f"   v: min={min(all_v):.3f}, max={max(all_v):.3f}, unique_values={len(set(all_v))}")
        print(f"   ω: min={min(all_omega):.3f}, max={max(all_omega):.3f}, unique_values={len(set(all_omega))}")
        print(f"   x: min={min(all_x):.3f}, max={max(all_x):.3f}, unique_values={len(set(all_x))}")
        print(f"   y: min={min(all_y):.3f}, max={max(all_y):.3f}, unique_values={len(set(all_y))}")
        print(f"   θ: min={min(all_theta):.3f}, max={max(all_theta):.3f}, unique_values={len(set(all_theta))}")
        
        # Check timestamps
        timestamps = list(camera_actions.keys())
        timestamps.sort()
        
        print(f"\n⏱️  TIMING ANALYSIS:")
        print(f"   Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"   Time gaps between frames:")
        
        gaps = []
        for i in range(1, min(10, len(timestamps))):  # Check first 10 gaps
            gap = timestamps[i] - timestamps[i-1]
            gaps.append(gap)
            print(f"     Frame {i}: {gap:.3f}s")
        
        if gaps:
            print(f"   Average gap: {sum(gaps)/len(gaps):.3f}s")
            print(f"   Expected FPS: {1/(sum(gaps)/len(gaps)):.1f}")
    
    # Diagnostic suggestions
    print(f"\n🔧 DIAGNOSTIC SUGGESTIONS:")
    print("-" * 40)
    
    if all(v == 0 for v in all_v) and all(omega == 0 for omega in all_omega):
        print("❌ All velocities are zero. Possible causes:")
        print("   1. Robot was not receiving goals from Goal Generator")
        print("   2. Robot was in a stopped/waiting state")
        print("   3. DWA planner was not finding valid paths")
        print("   4. Robot was waiting for recovery points")
        
    if all(x == 0 and y == 0 and theta == 0 for x, y, theta in zip(all_x, all_y, all_theta)):
        print("❌ All poses are (0,0,0). Possible causes:")
        print("   1. TF transform from 'map' to 'base_link' was not available")
        print("   2. Robot localization was not working")
        print("   3. Map frame was not properly set up")
        
    print(f"\n💡 TO GET MEANINGFUL DATA:")
    print("   1. Make sure the robot is receiving goals")
    print("   2. Check that TF transforms are being published")
    print("   3. Ensure the robot is actually moving during recording")
    print("   4. Check ROS topics: /move_base_simple/goal, /tf, /cmd_vel")

def find_latest_results_dir():
    """Find the most recent results directory"""
    base_dir = "/home/mrvik/dram_ws"
    result_dirs = []
    
    for item in os.listdir(base_dir):
        if item.startswith('dram_rosbag_results_'):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path):
                mtime = os.path.getmtime(full_path)
                result_dirs.append((full_path, mtime))
    
    if result_dirs:
        result_dirs.sort(key=lambda x: x[1], reverse=True)
        return result_dirs[0][0]
    return None

if __name__ == "__main__":
    # Find the latest results directory
    latest_dir = find_latest_results_dir()
    
    if latest_dir:
        camera_pickle = os.path.join(latest_dir, 'camera_actions.pkl')
        if os.path.exists(camera_pickle):
            debug_pickle_data(camera_pickle)
        else:
            print(f"❌ No camera_actions.pkl found in {latest_dir}")
    else:
        print("❌ No results directories found!")







