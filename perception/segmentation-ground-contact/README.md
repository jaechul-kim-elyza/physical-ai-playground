# Segmentation Ground Contact Estimation using RF-DETR

## Overview

This project explores **pixel-level ground contact estimation** for autonomous driving perception.

Traditional object detection approaches estimate object location using **bounding boxes**, but bounding boxes have fundamental limitations when estimating the actual contact point between an object and the road surface.

Examples:

- Vehicle bounding box bottom ≠ tire-road contact point
- Pedestrian bounding box bottom ≠ foot position
- Occlusion causes unstable bottom localization
- Perspective distortion makes box-based geometry inaccurate

This experiment investigates whether **instance segmentation masks** can provide more accurate ground-contact information by extracting the **bottom envelope of object masks**.

The pipeline:
