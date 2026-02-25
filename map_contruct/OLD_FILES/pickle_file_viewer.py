# import pickle
# import os
# import json
# from datetime import datetime

# def view_pickle_file(pickle_path):
#     """Comprehensive pickle file viewer"""
    
#     if not os.path.exists(pickle_path):
#         print(f"❌ File not found: {pickle_path}")
#         return
    
#     print(f"📂 Loading pickle file: {pickle_path}")
#     print(f"📏 File size: {os.path.getsize(pickle_path)} bytes")
#     print("=" * 80)
    
#     try:
#         with open(pickle_path, 'rb') as f:
#             data = pickle.load(f)
        
#         print("📊 PICKLE FILE STRUCTURE:")
#         print("-" * 40)
        
#         if isinstance(data, dict):
#             for key, value in data.items():
#                 print(f"🔑 Key: '{key}'")
#                 if isinstance(value, dict):
#                     print(f"   📋 Type: Dictionary with {len(value)} items")
#                     if len(value) > 0:
#                         first_key = next(iter(value.keys()))
#                         print(f"   📝 Sample key: {first_key}")
#                         print(f"   📝 Sample value: {value[first_key]}")
#                 elif isinstance(value, list):
#                     print(f"   📋 Type: List with {len(value)} items")
#                     if len(value) > 0:
#                         print(f"   📝 Sample item: {value[0]}")
#                 else:
#                     print(f"   📋 Type: {type(value).__name__}")
#                     print(f"   📝 Value: {value}")
#                 print()
        
#         # If this is a camera-action file, show detailed analysis
#         if 'camera_timestamp_to_action' in data:
#             print("📸 CAMERA-ACTION ANALYSIS:")
#             print("-" * 40)
            
#             camera_actions = data['camera_timestamp_to_action']
#             print(f"🎬 Total camera frames: {len(camera_actions)}")
            
#             if len(camera_actions) > 0:
#                 # Get timestamp range
#                 timestamps = list(camera_actions.keys())
#                 timestamps.sort()
                
#                 start_time = timestamps[0]
#                 end_time = timestamps[-1]
#                 duration = end_time - start_time
                
#                 print(f"⏱️  Time range: {duration:.2f} seconds")
#                 print(f"📅 Start time: {datetime.fromtimestamp(start_time)}")
#                 print(f"📅 End time: {datetime.fromtimestamp(end_time)}")
#                 print(f"🎯 Average FPS: {len(timestamps)/duration:.2f}")
#                 print()
                
#                 # Show first few entries
#                 print("📋 FIRST 5 CAMERA-ACTION PAIRS:")
#                 print("-" * 40)
#                 for i, (timestamp, action_data) in enumerate(list(camera_actions.items())[:5]):
#                     print(f"Frame {i+1}:")
#                     print(f"  📅 Timestamp: {timestamp}")
#                     print(f"  🚗 Linear velocity (v): {action_data['v']:.3f} m/s")
#                     print(f"  🔄 Angular velocity (ω): {action_data['omega']:.3f} rad/s")
#                     if 'robot_pose' in action_data:
#                         x, y, theta = action_data['robot_pose']
#                         print(f"  📍 Robot pose: ({x:.3f}, {y:.3f}, {theta:.3f})")
#                     print()
                
#                 # Show velocity statistics
#                 velocities = [action['v'] for action in camera_actions.values()]
#                 omegas = [action['omega'] for action in camera_actions.values()]
                
#                 print("📊 VELOCITY STATISTICS:")
#                 print("-" * 40)
#                 print(f"🚗 Linear velocity:")
#                 print(f"   Min: {min(velocities):.3f} m/s")
#                 print(f"   Max: {max(velocities):.3f} m/s")
#                 print(f"   Avg: {sum(velocities)/len(velocities):.3f} m/s")
#                 print(f"🔄 Angular velocity:")
#                 print(f"   Min: {min(omegas):.3f} rad/s")
#                 print(f"   Max: {max(omegas):.3f} rad/s")
#                 print(f"   Avg: {sum(omegas)/len(omegas):.3f} rad/s")
        
#         # If this is the main action log file
#         elif 'actions' in data and 'camera_actions' in data:
#             print("🔄 MAIN ACTION LOG ANALYSIS:")
#             print("-" * 40)
            
#             regular_actions = data['actions']
#             camera_actions = data['camera_actions']
            
#             print(f"🎬 Regular actions: {len(regular_actions)}")
#             print(f"📸 Camera-synced actions: {len(camera_actions)}")
            
#             if 'session_info' in data:
#                 session = data['session_info']
#                 print(f"⏱️  Session duration: {session.get('end_time', 0) - session.get('start_time', 0):.2f} seconds")
#                 print(f"🎯 Total camera frames: {session.get('total_camera_frames', 0)}")
        
#         print("\n✅ Pickle file loaded successfully!")
        
#     except Exception as e:
#         print(f"❌ Error loading pickle file: {e}")

# def find_latest_pickle_files():
#     """Find the most recent pickle files"""
#     base_dir = "/home/mrvik/dram_ws"
#     pickle_files = []
    
#     for root, dirs, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith('.pkl'):
#                 full_path = os.path.join(root, file)
#                 mtime = os.path.getmtime(full_path)
#                 pickle_files.append((full_path, mtime))
    
#     # Sort by modification time (newest first)
#     pickle_files.sort(key=lambda x: x[1], reverse=True)
    
#     print("🔍 RECENT PICKLE FILES:")
#     print("-" * 60)
#     for i, (file_path, mtime) in enumerate(pickle_files[:10]):  # Show top 10
#         mod_time = datetime.fromtimestamp(mtime)
#         file_size = os.path.getsize(file_path)
#         print(f"{i+1:2d}. {os.path.basename(file_path)}")
#         print(f"    📂 {file_path}")
#         print(f"    📅 {mod_time}")
#         print(f"    📏 {file_size} bytes")
#         print()
    
#     return [fp for fp, _ in pickle_files]

# if __name__ == "__main__":
#     print("🥒 PICKLE FILE VIEWER")
#     print("=" * 80)
    
#     # Find recent pickle files
#     recent_files = find_latest_pickle_files()
    
#     if recent_files:
#         print("\n" + "=" * 80)
#         print("📸 VIEWING MOST RECENT CAMERA-ACTION FILE:")
#         print("=" * 80)
        
#         # Look for camera_actions.pkl files first
#         camera_files = [f for f in recent_files if 'camera_actions.pkl' in f]
#         if camera_files:
#             view_pickle_file(camera_files[0])
#         else:
#             # Fallback to most recent pickle file
#             view_pickle_file(recent_files[0])
#     else:
#         print("❌ No pickle files found!")
        
#         # Try the specific path from your original code
#         specific_path = '/home/mrvik/dram_ws/dram_rosbag_results_20250915_125435/camera_actions.pkl'
#         if os.path.exists(specific_path):
#             print(f"\n📂 Found specific file: {specific_path}")
#             view_pickle_file(specific_path)

import pickle
import os
import glob

def check_camera_actions_format(file_path):
    """Check the format of a camera_actions.pkl file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old and new formats
        if isinstance(data, dict):
            if 'camera_timestamp_to_action' in data:
                camera_actions = data['camera_timestamp_to_action']
                format_type = "OLD FORMAT (nested)"
            else:
                camera_actions = data
                format_type = "NEW FORMAT (direct)"
        else:
            camera_actions = data
            format_type = "UNKNOWN FORMAT"
        
        # Check first item structure
        if camera_actions:
            first_timestamp = next(iter(camera_actions.keys()))
            first_action = camera_actions[first_timestamp]
            
            has_v = 'v' in first_action
            has_omega = 'omega' in first_action
            expected_format = has_v and has_omega
            
            print(f"📂 {os.path.basename(file_path)}")
            print(f"   📊 Type: {type(data).__name__}")
            print(f"   🔧 Format: {format_type}")
            print(f"   📏 Length: {len(camera_actions)}")
            print(f"   ✅ Has v & omega: {expected_format}")
            print(f"   📝 Keys: {list(first_action.keys())}")
            print(f"   📋 Sample: v={first_action.get('v', 'N/A')}, omega={first_action.get('omega', 'N/A')}")
            print()
            
            return expected_format
        else:
            print(f"📂 {os.path.basename(file_path)} - EMPTY FILE")
            return False
            
    except Exception as e:
        print(f"❌ {os.path.basename(file_path)} - ERROR: {e}")
        return False

# Find all camera_actions.pkl files
base_dir = "/home/mrvik/dram_ws"
camera_files = []

# Search for camera_actions.pkl files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == 'camera_actions.pkl':
            camera_files.append(os.path.join(root, file))

# Also check Downloads folder
downloads_files = glob.glob("/home/mrvik/dram_ws/mppi_results_20250915_213453/camera_actions.pkl")
camera_files.extend(downloads_files)

print("🔍 CHECKING ALL CAMERA_ACTIONS.PKL FILES")
print("=" * 60)

if not camera_files:
    print("❌ No camera_actions.pkl files found!")
else:
    valid_count = 0
    for file_path in camera_files:
        is_valid = check_camera_actions_format(file_path)
        if is_valid:
            valid_count += 1
    
    print("=" * 60)
    print(f"📊 SUMMARY: {valid_count}/{len(camera_files)} files have correct format")
    print(f"✅ Expected format: {{timestamp: {{'v': float, 'omega': float}}}}")
        