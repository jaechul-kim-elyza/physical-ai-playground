FROM osrf/ros:jazzy-desktop

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-turtlebot3* \
    ros-jazzy-realsense2-camera \
    python3-pip \
    vim \
    tree \
 && rm -rf /var/lib/apt/lists/*
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo 'if [ -f /root/ros2_ws/install/setup.bash ]; then source /root/ros2_ws/install/setup.bash; fi' >> /root/.bashrc