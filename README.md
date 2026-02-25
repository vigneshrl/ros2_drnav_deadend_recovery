# DRaM Navigation Package

**DRaM (Dead-end Risk-aware Mapping)** is a ROS2 navigation package that uses RGB-LiDAR fusion to detect dead-ends proactively and enable intelligent recovery in unmapped environments.

## 📋 Table of Contents

1. [Package Structure](#package-structure)
2. [Core Components](#core-components)
3. [How It Works](#how-it-works)
4. [Running Different Scenes](#running-different-scenes)
5. [Launch Files](#launch-files)
6. [Metrics & Logging](#metrics--logging)
7. [Configuration](#configuration)

---

## 📁 Package Structure

```
map_contruct/
├── baselines/              # Baseline planners for comparison
│   ├── dwa/               # Dynamic Window Approach planner
│   ├── nav2_dwb/          # Nav2 DWB (Dynamic Window Behavior) planner
│   └── mppi/              # Model Predictive Path Integral planner
│
├── goal_generator/        # Unified waypoint generator
│   └── goal_generator.py  # EDE-based goal generation (Score = J_geom + λ·EDE)
│
├── scripts/
│   ├── inference/         # DRaM inference node
│   │   └── infer_vis.py   # Multi-camera + LiDAR dead-end detection
│   │
│   ├── models/            # Neural network models
│   │   └── model_CA.py    # Cross-attention fusion model
│   │
│   ├── utilities/         # Supporting nodes
│   │   ├── recovery_points.py        # Recovery point manager
│   │   ├── cost_layer_processor.py   # Bayesian costmap updates
│   │   ├── dram_heatmap_viz.py       # Risk map visualization
│   │   ├── pointcloud_segmenter.py   # LiDAR segmentation
│   │   └── ...
│   │
│   └── viz/              # Visualization & analysis tools
│       ├── evaluation_framework.py
│       ├── method_comparison_analyzer.py
│       └── ...
│
├── launch/                # Launch files
│   ├── dwa_goal_generator.launch.py
│   ├── mppi_goal_generator.launch.py
│   ├── inference_dwa.launch.py
│   └── ...
│
└── results/               # All results, logs, and metrics
    ├── inference_results/
    ├── scene_metrics/
    └── logs/
```

---

## 🔧 Core Components

### 1. **Inference Node** (`scripts/inference/infer_vis.py`)

**Purpose**: Multi-camera + LiDAR fusion for dead-end detection

**Key Features**:
- Processes 3 camera views (front, left, right) + 3 LiDAR views
- Uses cross-attention neural network (`model_CA.py`)
- Outputs:
  - `/dead_end_detection/path_status` - 3-directional path probabilities [front, left, right]
  - `/dead_end_detection/is_dead_end` - Binary dead-end flag
  - `/dead_end_detection/recovery_points` - Recovery point locations

**Topics**:
- Subscribes: `/argus/ar0234_*/image_raw`, `/lidar/*/points`
- Publishes: `/dead_end_detection/*`

---

### 2. **Goal Generator** (`goal_generator/goal_generator.py`)

**Purpose**: Unified waypoint generation using EDE (Exposure to Dead-End) scoring

**Algorithm**: `Score(θ) = J_geom(θ) + λ·EDE(θ)`

- **J_geom(θ)**: Geometric costs (collision, feasibility, smoothness, range bias)
- **EDE(θ)**: Semantic risk from DRaM (only when λ > 0)
- **λ**: Method-specific weight
  - DRaM methods: λ = 1.0 (uses semantic risk)
  - LiDAR baselines: λ = 0.0 (pure geometric)

**Features**:
- Samples 36 headings θ ∈ [-π, π]
- Generates waypoints 3-5m ahead
- Recovery mode when all 3 directions blocked
- Rate: 5-10 Hz

**Topics**:
- Subscribes: `/dram_exploration_map`, `/dead_end_detection/*`, `/map`
- Publishes: `/move_base_simple/goal`

---

### 3. **Recovery Mechanism** (`scripts/utilities/recovery_points.py`)

**Purpose**: Store and manage recovery points for dead-end recovery

**How It Works**:
1. **Detection**: When ≥1 path is open, location is stored as recovery point
2. **Ranking**: Recovery points ranked by number of open paths (1-3)
3. **Triggering**: When all 3 directions blocked (dead-end), robot:
   - Stops forward motion
   - Looks up nearest/best recovery point
   - Sets recovery point as new Nav2 goal
   - Executes smooth U-turn

**Recovery Point Types**:
- Type 1: 1 open path (light blue)
- Type 2: 2 open paths (dark blue)
- Type 3: 3+ open paths (purple)

---

### 4. **Bayesian Costmap Updates** (`scripts/utilities/cost_layer_processor.py`)

**Purpose**: Update semantic costmap from "Blocked" (static map) to "Open" (live sensors)

**Algorithm**: Bayesian Log-Odds Update
```
log_odds_new = log_odds_old + log(p / (1-p))
```

**Process**:
1. Receives path probabilities from inference node
2. Converts to binary using 0.56 threshold
3. Updates costmap cells using Bayesian log-odds
4. Overrides static map when camera detects open doorway

**Visualization**:
- Green = Open/Safe (path probability > 0.56)
- Red = Blocked/Dead-end (path probability ≤ 0.56)
- Blue/Purple = Recovery points

**Topics**:
- Subscribes: `/dead_end_detection/path_status`, `/map`
- Publishes: `/cost_layer`, `/dead_end_detection/recovery_points`

---

### 5. **Baseline Planners**

#### **DWA Planner** (`baselines/dwa/dwa_planner.py`)
- Dynamic Window Approach
- Samples velocity space (v, ω)
- Optimizes for goal distance + safety

#### **Nav2 DWB Planner** (`baselines/nav2_dwb/nav2_dwb_planner.py`)
- Nav2 Dynamic Window Behavior
- Forward simulation with acceleration limits
- Path distance + goal distance scoring

#### **MPPI Planner** (`baselines/mppi/mppi_planner.py`)
- Model Predictive Path Integral
- Trajectory sampling with noise
- Softmax-weighted control update

---

## 🎯 How It Works

### **Complete DRaM Pipeline**:

```
1. Sensors → Inference Node
   ├── 3 Cameras (RGB) → Image Encoder (ResNet50)
   └── 3 LiDAR → Point Cloud Encoder (PointNet)
   
2. Cross-Attention Fusion
   ├── Image → LiDAR attention
   └── LiDAR → Image attention
   
3. Outputs
   ├── Path Status [front, left, right] probabilities
   ├── Dead-end flag
   └── Direction vectors
   
4. Costmap Processor
   ├── Bayesian log-odds update
   └── Recovery point detection
   
5. Goal Generator
   ├── Sample headings θ
   ├── Compute Score(θ) = J_geom + λ·EDE
   └── Publish waypoint to /move_base_simple/goal
   
6. Planner (DWA/MPPI/Nav2 DWB)
   ├── Receive goal
   ├── Plan trajectory
   └── Publish /cmd_vel
   
7. Recovery (if dead-end detected)
   ├── Stop forward motion
   ├── Lookup recovery point
   └── Navigate to recovery point
```

---

## 🎬 Running Different Scenes

### **Scene 1: Indoor (Unmapped) - Recovery Process**

**Goal**: Demonstrate recovery mechanism in T-junction dead-end

**Setup**:
```bash
# Launch DRaM with DWA
ros2 launch map_contruct inference_dwa.launch.py

# Or with goal generator
ros2 launch map_contruct dwa_goal_generator.launch.py
```

**Tasks**:
1. Drive robot into dead-end until EDE is high
2. Force recovery (all 3 directions blocked)
3. Robot stops, looks up recovery point, executes U-turn
4. Record metrics: Recovery Success Rate, Path Smoothness

**Metrics to Collect**:
- Path Smoothness (curvature/jerkiness of U-turn)
- Recovery Success Rate
- Time to recovery point
- Comparison: Point-Return vs. Spin-In-Place

**Logging**:
```bash
# Record rosbag with required topics
ros2 bag record /odom /cmd_vel /dead_end_detection/is_dead_end \
  /dead_end_detection/recovery_points /move_base_simple/goal
```

---

### **Scene 2: Lab Hallway (Dynamic Re-entry) - Static Map**

**Goal**: Demonstrate Bayesian costmap updates override static map

**Setup**:
1. **Generate static map** with door fully closed
   ```bash
   # Map should show solid wall at entrance
   ```

2. **Launch baseline** (Nav2 DWB, DWA, or MPPI)
   ```bash
   ros2 launch map_contruct dwa_goal_generator.launch.py
   # Set goal inside lab
   ```

3. **Open door completely** while robot approaches

4. **Launch DRaM method**
   ```bash
   ros2 launch map_contruct inference_dwa.launch.py
   ```

**Expected Behavior**:
- **Baselines**: Fail to create path (static map shows wall) or hover outside door
- **DRaM**: RGB-LiDAR fusion recognizes open doorway → Bayesian update changes costmap from "Blocked" to "Open" → Robot enters

**Metrics to Collect**:
- Time-to-Goal (baselines: ∞, DRaM: finite)
- Re-classification Latency (camera sees door → path changes)
- Success Rate (how many times each planner enters)
- Semantic Map Visualization (screenshot: Grey wall → Green open)

**Visualization**:
```bash
# View costmap in RViz
# Topic: /dram_exploration_map
# Look for: Grey/Black (static map) → Green (semantic update)
```

---

### **Scene 3: Outdoor (Unmapped) - Proactive Avoidance**

**Goal**: Detect dead-end visually before LiDAR sees closing/end

**Setup**:
```bash
# Launch baseline
ros2 launch map_contruct mppi_goal_generator.launch.py
# Set global goal 10m past end of path

# Launch DRaM
ros2 launch map_contruct inference_dwa.launch.py
```

**Expected Behavior**:
- **Baselines**: Drive until -1m from end (LiDAR range limit)
- **DRaM**: React at -5m+ (camera range, proactive detection)

**Metrics to Collect**:
- **PAD (Pre-emptive Avoidance Distance)**: Distance from robot to wall when it first stops/turns
- **Negative Progress**: Total distance traveled into dead-end before retreating
- **Outcome**: Baselines (-1m), DRaM (-5m+)

**Debugging**:
- If robot doesn't turn: Check EDE threshold and Bayesian update rate
- Adjust `lambda_ede` parameter in goal generator

---

## 🚀 Launch Files

### **DRaM Method Launches**

#### `inference_dwa.launch.py`
Launches inference node + DWA planner for DRaM navigation
```bash
ros2 launch map_contruct inference_dwa.launch.py
```

#### `dwa_goal_generator.launch.py`
Launches DWA planner + Goal Generator (for baselines or DRaM)
```bash
ros2 launch map_contruct dwa_goal_generator.launch.py
```

#### `mppi_goal_generator.launch.py`
Launches MPPI planner + Goal Generator
```bash
ros2 launch map_contruct mppi_goal_generator.launch.py
```

### **Goal Generator Standalone**
```bash
ros2 launch map_contruct goal_generator.launch.py \
  method:=multi_camera_dram \
  rate:=7.0 \
  horizon:=4.0
```

**Parameters**:
- `method`: `multi_camera_dram`, `single_camera_dram`, `dwa_lidar`, `mppi_lidar`
- `rate`: Goal generation rate in Hz (5-10 recommended)
- `horizon`: Horizon distance in meters (3-5m recommended)

---

## 📊 Metrics & Logging

### **Required Topics for Logging**

For every run, record a `.bag` or `.mcap` file with:

```bash
ros2 bag record \
  /odom                    # Distance traveled
  /cmd_vel                 # Control commands (smoothness)
  /dead_end_detection/is_dead_end
  /dead_end_detection/path_status
  /dead_end_detection/recovery_points
  /move_base_simple/goal
  /dram_exploration_map    # Semantic costmap
  /cost_layer              # Bayesian updates
```

### **Metrics Collected**

Each planner node collects:
- `total_distance`: Total distance traveled
- `total_energy`: Energy consumption
- `dead_end_detections`: Number of dead-ends detected
- `recovery_activations`: Number of recovery actions
- `completion_time`: Time to reach goal (or ∞ if failed)
- `time_trapped`: Time spent in dead-ends

**Output Location**: `results/scene_metrics/`

### **Scene-Specific Metrics**

#### Scene 1 (Recovery):
- Recovery Success Rate
- Path Smoothness (curvature)
- Point-Return vs. Spin-In-Place comparison

#### Scene 2 (Dynamic Re-entry):
- Time-to-Goal (baselines: ∞, DRaM: finite)
- Re-classification Latency
- Success Rate
- Semantic Map Visualization

#### Scene 3 (Proactive Avoidance):
- PAD (Pre-emptive Avoidance Distance)
- Negative Progress
- Detection distance comparison

---

## ⚙️ Configuration

### **Goal Generator Parameters**

Edit launch file or use parameters:
```python
parameters=[{
    'method_type': 'multi_camera_dram',  # or 'dwa_lidar', 'mppi_lidar'
    'lambda_ede': 1.0,                    # EDE weight (0 for baselines)
    'goal_generation_rate': 7.0,          # Hz
    'horizon_distance': 4.0               # meters
}]
```

### **Inference Node Parameters**

```python
parameters=[{
    'robot_mode': True,                   # Optimize for real robot
    'save_visualizations': True           # Save inference visualizations
}]
```

### **Recovery Threshold**

In `goal_generator.py`:
```python
self.path_blocked_threshold = 0.56  # Threshold for path blocked
self.recovery_threshold = 4         # Consecutive dead-ends before recovery
```

### **Model Weights**

Update path in `scripts/inference/infer_vis.py`:
```python
model_path = '/home/mrvik/dram_ws/model_wts/model_best.pth'
```

---

## 🔬 Ablation Study

To run ablation study (without cross-attention):

1. Modify `scripts/models/model_CA.py` to disable cross-attention
2. Use simple concatenation: `torch.cat([img_feats, lidar_feats], dim=1)`
3. Retrain or use modified model
4. Compare metrics with full DRaM method

---

## 📝 Notes

### **MPPI Configuration**
- Use default critics
- Use "Standard" version for testing
- Ensure proper noise parameters

### **Static Map Creation**
- Generate map with door fully closed
- Costmap should show solid wall at entrance
- Use SLAM toolbox or manual mapping

### **Visualization**
- RViz: View `/dram_exploration_map` for risk visualization
- Green = Safe, Red = Dead-end, Blue/Purple = Recovery points
- Check `/cost_layer` for Bayesian updates

---

## 🐛 Troubleshooting

### **Robot doesn't turn in Scene 3**
- Check EDE threshold in goal generator
- Verify Bayesian update rate
- Check camera/LiDAR topics are publishing

### **Recovery not triggering**
- Verify all 3 directions are blocked (check `/dead_end_detection/path_status`)
- Check recovery point threshold
- Ensure recovery points are being published

### **Static map not updating (Scene 2)**
- Verify cost_layer_processor is running
- Check Bayesian update is receiving path_status
- Verify robot position in map frame

---

## 📚 References

- **EDE Scoring**: `Score(θ) = J_geom(θ) + λ·EDE(θ)`
- **Bayesian Update**: Log-odds formulation in `cost_layer_processor.py`
- **Recovery Strategy**: Point-return mechanism in `recovery_points.py`

---

## 👥 Contributors

- Gershom: Static map creation, Nav2 DWB, DWA, MPPI testing, Ablation study
- Pon Ashwin: Nav2 DWB, DWA, MPPI code debug, Ablation study
- Vignesh: Goal Generator code debug, Model fine-tuning, Paper corrections

---

**Last Updated**: Based on Work Plan for Scenes 1, 2, and 3

