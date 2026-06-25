# DR.Nav — Dead-End Risk-Aware Navigation (v1.0.0)

ROS 2 navigation system that uses RGB-LiDAR sensor fusion and deep learning to predict dead ends and recover from them autonomously.

```
[Sensors] → [Inference] → [Risk Mapping] → [Goal Generation] → [Planner] → [Robot]
```

---

## Installation

You can run DR.Nav either **natively** or in **Docker** (recommended for deployment).

### Option A — Native Install

**Prerequisites:** ROS 2 Humble, Python 3.10, NVIDIA GPU with CUDA, `slam_toolbox`, model weights ([download here](https://drive.google.com/file/d/1pf3I-CjcveE9MYK_6e_c95H-N6_k_1Kr/view?usp=sharing)). This is the weights for the single camera model. 

```bash
# 1. Clone into your ROS 2 workspace
cd ~/ros2_ws/src
git clone <repository-url> ros2_drnav_deadend_recovery

# 2. Install Python dependencies (use system Python, not conda — ROS 2 nodes use /usr/bin/python3)
/usr/bin/python3 -m pip install torch torchvision numpy opencv-python pillow matplotlib scipy

# 3. Install ROS dependencies
sudo apt install ros-humble-slam-toolbox ros-humble-sensor-msgs-py ros-humble-pointcloud-to-laserscan

# 4. Build
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

### Option B — Docker Install (recommended)

Docker bundles everything (ROS 2 Humble, PyTorch + CUDA, all Python packages). No local Python or ROS install needed.

**Prerequisites:** Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

#### 1. Build the image

```bash
cd ros2_drnav_deadend_recovery
docker build -t drnav .
```

This builds a container with ROS 2 Humble, PyTorch (CUDA 11.8), and the full workspace pre-compiled.

#### 2. Start a container

```bash
docker run --rm -it --gpus all \
  --network host --ipc host \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  --name drnav \
  -v /path/to/your/model_wts:/model_wts \
  drnav bash
```

| Flag | Why |
|---|---|
| `--gpus all` | GPU access for PyTorch inference |
| `--network host` | ROS 2 DDS discovery between container and robot/sim |
| `--ipc host` | FastDDS shared-memory transport (without this, topics are visible but carry no data) |
| `-v /path/to/model_wts:/model_wts` | Mount your model weights into the container |

Once inside, everything is already sourced — you can run launch commands immediately.

#### 3. Open more terminals into the same container

```bash
docker exec -it drnav bash
```

All `exec` sessions inherit the network and IPC settings automatically.

#### 4. Copy files into/out of a running container

```bash
# Copy a file from host → container
docker cp /host/path/to/model_best.pth drnav:/model_wts/model_best.pth

# Copy a file from container → host
docker cp drnav:/ros2_ws/src/map_contruct/scripts/models/model_CA.py ./

# Copy an entire folder into the container
docker cp /host/path/to/my_folder/ drnav:/ros2_ws/src/map_contruct/
```

#### 5. Update code inside the container

If you edit code on the host and want the container to reflect it, you have two options:

**Mount the repo as a volume (live edits, no image rebuild):**

```bash
docker run --rm -it --gpus all \
  --network host --ipc host \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  --name drnav \
  -v /path/to/your/model_wts:/model_wts \
  -v /path/to/ros2_drnav_deadend_recovery:/ros2_ws/src/map_contruct \
  drnav bash
```

This mounts your local repo directly. Any file you edit on the host is instantly visible inside the container. After editing, rebuild inside:

```bash
cd /ros2_ws && colcon build --symlink-install && source install/setup.bash
```

**Or rebuild the image from scratch (clean, reproducible):**

```bash
docker build -t drnav .
```

#### 6. Save a modified container as a new image

If you installed extra packages or made changes inside a running container:

```bash
# From the host (while container is running)
docker commit drnav drnav:v1.1

# Start from the saved image later
docker run --rm -it --gpus all --network host --ipc host \
  --name drnav -v /path/to/model_wts:/model_wts drnav:v1.1 bash
```

#### 7. Fix: camera topics drop when LiDAR is running

Large PointCloud2 messages can overflow the default UDP buffer. Set this on the **host** (not inside Docker):

```bash
sudo sysctl -w net.core.rmem_max=67108864
sudo sysctl -w net.core.rmem_default=67108864
sudo sysctl -w net.core.wmem_max=67108864
```

Make permanent:

```bash
echo "net.core.rmem_max=67108864" | sudo tee -a /etc/sysctl.conf
echo "net.core.rmem_default=67108864" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max=67108864" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## Running

### Mapless Mode (exploration with SLAM)

**Terminal 1 — Localisation (SLAM):**

```bash
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=/path/to/slam_params.yaml \
  use_sim_time:=false
```

Or with default parameters:

```bash
ros2 launch slam_toolbox online_async_launch.py
```

> SLAM provides the `/map` topic and `map → odom` TF. It requires a `/scan` topic — if your LiDAR only publishes PointCloud2, run `pointcloud_to_laserscan` first (included in the DR.Nav launch when using `method:=dram`).

**Terminal 2 — DR.Nav:**

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch map_contruct mapless.launch.py method:=dram model_path:=/path/to/model_best.pth
```

**Baselines (no model needed):**

```bash
ros2 launch map_contruct mapless.launch.py method:=dwa
ros2 launch map_contruct mapless.launch.py method:=mppi
ros2 launch map_contruct mapless.launch.py method:=nav2_dwb
```

**With RViz and bag recording:**

```bash
ros2 launch map_contruct mapless.launch.py method:=dram \
  model_path:=/path/to/model_best.pth use_rviz:=true record:=true run_id:=1
```

---

### Map-Based Mode (pre-built map with Nav2)

**Terminal 1 — Nav2:**

```bash
ros2 launch nav2_bringup bringup_launch.py \
  map:=/path/to/your_map.yaml \
  params_file:=/path/to/nav2_params.yaml
```

**Terminal 2 — DR.Nav perception (only needed for `dram`):**

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch map_contruct map_based.launch.py method:=dram
```

Nav2 handles driving. DR.Nav adds the risk perception layer on top. Baselines (`dwa`, `mppi`, `nav2_dwb`) are handled entirely by Nav2 — no second terminal needed.

---

### Single Camera Mode

If you only have one camera + one LiDAR, use the single-view model:

```bash
ros2 launch map_contruct mapless.launch.py method:=dram \
  model_path:=/path/to/model_single_view_best.pth single_camera:=true
```

Or run the node directly:

```bash
ros2 run map_contruct infer_vis --ros-args \
  -p single_camera:=true \
  -p model_path:=/path/to/model_single_view_best.pth \
  -p robot_mode:=true
```

In single-camera mode, the node only subscribes to front camera and front LiDAR topics.

---

## Architecture

| Stage | Node | Role |
|---|---|---|
| Sensor preprocessing | `pointcloud_segmenter` | Splits omnidirectional LiDAR into directional sectors |
| Dead-end inference | `infer_vis` | Runs the DRaM model; outputs path probabilities per direction |
| Risk mapping | `dram_risk_map` | Builds a Bayesian safety grid; tracks recovery waypoints |
| Goal generation | `goal_generator` | Scores candidate headings; sends waypoints to the planner |
| Local planning | `dwa_planner` | Executes velocity commands toward the current waypoint |

Unified scoring formula:

```
Score(θ) = J_geom(θ) + λ · EDE(θ)
```

- `λ = 1.0` for DR.Nav, `λ = 0.0` for all baselines

---

## TF Frames

```
map → odom → body (base_link)
```

| Transform | Provider |
|---|---|
| `map → odom` | `slam_toolbox` (mapless) or AMCL via `nav2_bringup` (map-based) |
| `odom → body` | `odom_tf_broadcaster` (included in launch files) |

Ensure your hardware publishes `/odom_lidar` (nav_msgs/Odometry) before launching.

---

## Key Parameters

| Parameter | Node | Default | Description |
|---|---|---|---|
| `model_path` | `infer_vis` | `$MODEL_PATH` or `/model_wts/model_best.pth` | Path to model weights |
| `robot_mode` | `infer_vis` | `true` | RELIABLE QoS for cameras, BEST_EFFORT for LiDAR, 5 Hz |
| `single_camera` | `infer_vis` | `false` | Use single front camera + LiDAR only |
| `save_visualizations` | `infer_vis` | `false` | Save per-frame PNGs and JSON to disk |
| `method_type` | `goal_generator` | `multi_camera_dram` | Selects EDE scoring and recovery logic |
| `lambda_ede` | `goal_generator` | `1.0` | EDE weight (0 = baseline, no risk avoidance) |
| `horizon_distance` | `goal_generator` | `4.0` | Look-ahead ray length (metres) |

---

## ROS 2 Topics

**Inputs (from hardware):**

| Topic | Type |
|---|---|
| `/argus/ar0234_front_left/image_raw` | sensor_msgs/Image |
| `/argus/ar0234_side_left/image_raw` | sensor_msgs/Image |
| `/argus/ar0234_side_right/image_raw` | sensor_msgs/Image |
| `/os_cloud_node/points` | sensor_msgs/PointCloud2 |
| `/odom_lidar` | nav_msgs/Odometry |

**Internal (DR.Nav pipeline):**

| Topic | Type | Publisher |
|---|---|---|
| `/lidar/front/points` | PointCloud2 | `pointcloud_segmenter` |
| `/lidar/left/points` | PointCloud2 | `pointcloud_segmenter` |
| `/lidar/right/points` | PointCloud2 | `pointcloud_segmenter` |
| `/dead_end_detection/path_status` | Float32MultiArray | `infer_vis` |
| `/dead_end_detection/is_dead_end` | Bool | `infer_vis` |
| `/dram_exploration_map` | MarkerArray | `dram_risk_map` |
| `/dead_end_detection/recovery_points` | Float32MultiArray | `dram_risk_map` |
| `/move_base_simple/goal` | PoseStamped | `goal_generator` |
| `/cmd_vel` | Twist | `dwa_planner` |

**RViz visualization:**

| Topic | What it shows |
|---|---|
| `/dram_exploration_map` | Risk heatmap (green=safe, red=dead-end risk) |
| `/cost_layer` | Directional sector overlay at robot position |
| `/goal_generator/rays` | Scored heading rays |
| `/map` | Occupancy grid from SLAM or Nav2 |

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Cameras | 3× Argus AR0234 (or 1× in single-camera mode) |
| LiDAR | Ouster OS-series (omnidirectional, `/os_cloud_node/points`) |
| Odometry | `/odom_lidar` (nav_msgs/Odometry) |
| GPU | NVIDIA GPU with CUDA |

---

## Package Structure

```
ros2_drnav_deadend_recovery/
├── map_contruct/
│   ├── scripts/
│   │   ├── inference/
│   │   │   └── infer_vis.py              # DRaM model inference node
│   │   ├── models/
│   │   │   └── model_CA.py               # DeadEndDetectionModel (EfficientNet-B0 + PointNet + CrossAttention)
│   │   ├── control/
│   │   │   └── direct_vel_controller.py  # Direct velocity controller for DR.Nav
│   │   └── utilities/
│   │       ├── pointcloud_segmenter.py   # LiDAR sector splitter
│   │       ├── odom_tf_brodcaster.py     # Odometry → TF broadcaster
│   │       ├── dram_risk_map.py          # Bayesian risk grid + recovery manager
│   │       └── recovery_points.py        # RecoveryPointManager class
│   ├── goal_generator/
│   │   └── goal_generator.py             # Unified scoring and waypoint selection
│   └── baselines/
│       ├── dwa/dwa_planner.py
│       ├── mppi/mppi_planner.py
│       └── nav2_dwb/nav2_dwb_planner.py
├── launch/
│   ├── mapless.launch.py                 # Mapless mode (all methods)
│   └── map_based.launch.py              # Map-based mode (all methods)
├── config/
│   └── nav2_params.yaml
├── isaac_sim/                            # Isaac Sim 4.5 sensor setup scripts
├── Dockerfile
├── setup.py
└── package.xml
```

---

## Model Weights

Download from [Google Drive](https://drive.google.com/drive/folders/1WI5vdguuyMMoQxnnhyEcjAxYb8t-mCr_?usp=sharing).

| File | Use case |
|---|---|
| `model_best.pth` | 3-camera mode (front + left + right) |
| `model_single_view_best.pth` | Single-camera mode (front only) |

The model path is resolved in this order:
1. ROS parameter `model_path` (if set at launch)
2. Environment variable `MODEL_PATH`
3. Default: `/model_wts/model_best.pth`
