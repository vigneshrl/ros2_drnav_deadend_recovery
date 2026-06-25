FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV ROS_DISTRO=humble

# ── 1. ROS 2 Humble ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-slam-toolbox \
    ros-humble-topic-tools \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-visualization-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf2-sensor-msgs \
    ros-humble-nav2-msgs \
    ros-humble-sensor-msgs-py \
    ros-humble-pointcloud-to-laserscan \
    python3-colcon-common-extensions \
    python3-pip \
    nano vim \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Python ML packages ─────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    "numpy<2" \
    opencv-python-headless \
    Pillow \
    matplotlib \
    scipy

# ── 3. Build the ROS 2 workspace ─────────────────────────────────────────────
WORKDIR /ros2_ws/src
COPY . map_contruct/

WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

# Pre-generate slam params for real Jackal deployment
RUN cp /opt/ros/humble/share/slam_toolbox/config/mapper_params_online_async.yaml /ros2_ws/slam_params_real.yaml \
    && sed -i 's|^    use_sim_time:.*|    use_sim_time: false|' /ros2_ws/slam_params_real.yaml \
    && sed -i 's|^    scan_topic:.*|    scan_topic: /j100_0893/sensors/lidar3d_0/scan|' /ros2_ws/slam_params_real.yaml \
    && sed -i 's|^    map_frame:.*|    map_frame: map|' /ros2_ws/slam_params_real.yaml \
    && sed -i 's|^    odom_frame:.*|    odom_frame: 93/odom|' /ros2_ws/slam_params_real.yaml \
    && sed -i 's|^    base_frame:.*|    base_frame: 93/base_link|' /ros2_ws/slam_params_real.yaml

# ── 4. Source workspace on every shell ───────────────────────────────────────
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Mount model weights here at runtime:
#   docker run -v /your/path/model_wts:/model_wts drnav bash
RUN mkdir -p /model_wts
ENV MODEL_PATH=/model_wts/model_best.pth
