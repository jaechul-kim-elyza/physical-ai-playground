# Physical AI Playground

A research playground for **Physical AI**, focusing on vision-based perception, 
robotics, and real-world AI systems.

This repository contains experimental implementations for:
- Vision-based perception
- 3D geometry understanding
- Robot perception
- Simulation environments
- Real-time AI deployment

The goal is to bridge the gap between **AI models and physical-world interaction**.

## Repository structure

The workspace is organized into four main areas:

- docker/ros2-jazzy/: ROS 2 Jazzy development environment and container configuration
- ros2_ws/src/: ROS 2 packages for robotics workflows
  - tutorial_pkg
  - camera_receiver
  - rf_detr_ros
  - ackermann_controller
- perception/: perception models, experiments, and related tooling
- simulation/: simulation assets and environment setup

A typical layout looks like this:

```text
physical-ai-playground/
├── docker/
│   └── ros2-jazzy/
├── ros2_ws/
│   └── src/
│       ├── tutorial_pkg
│       ├── camera_receiver
│       ├── rf_detr_ros
│       └── ackermann_controller
├── perception/
└── simulation/
```

---

## Overview

Modern Physical AI systems require robust perception that can understand:

- What exists in the environment?
- Where are objects located?
- How does the robot interact with the scene?
- How can visual information be converted into actionable physical understanding?

This repository explores practical solutions combining:
