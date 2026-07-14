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

1. Runs RF-DETR instance segmentation on a nuScenes camera image.
2. Extracts the mask bottom envelope and groups its lowest/local-lowest runs as
   visible ground-contact candidates.
3. Projects those candidates onto the calibrated nuScenes ground plane and
   marks the contact nearest to the ego vehicle.
4. For vehicles with at least two reliable contacts, estimates a class-prior
   3D box, fits its projection to the segmentation silhouette, and renders
   only the face(s) visible to the ego camera.
5. Detects truncation and occlusion from image boundaries, predicted mask
   overlap, and detector-box overlap.  A 3D face is intentionally skipped for
   `occluded`, `possibly_occluded`, and `truncated` objects.

## Visible-face output

The script does not render a complete wireframe cuboid.  It renders the
physical surface visible from the ego viewpoint:

- Side/front/rear view: one vertical face.
- Oblique view: the dominant face plus an adjacent face when the secondary
  viewing component is material (for example, `RIGHT + REAR`).
- No reliable metric face, or an occluded/truncated object: the exact 2D
  segmentation exterior is shown instead of a speculative 3D face.

For a poor initial wheel-line pose, the segmentation fitting stage uses a
wider recovery search and then selects the smallest projected box that
contains the object contour.  The output includes the resulting
`contour_coverage` and visible-face mask metrics.

## Run

The nuScenes mini dataset path is currently configured in
`rfdetr_nuscenes_bottom_envelope.py` as `DATA_ROOT`.

```bash
cd perception/segmentation-ground-contact

# Default: first CAM_FRONT sample in nuScenes mini.
../../.venv/bin/python rfdetr_nuscenes_bottom_envelope.py

# A selected side-camera sample.  Outputs are written to OUTPUT_DIR.
MPLCONFIGDIR=/tmp/matplotlib \
NUSCENES_CAMERA_CHANNEL=CAM_FRONT_RIGHT \
NUSCENES_SAMPLE_TOKEN=ac452a60e8b34a7080c938c904b23057 \
OUTPUT_DIR=batch_side_vehicle_3d/sample_10 \
../../.venv/bin/python rfdetr_nuscenes_bottom_envelope.py
```

Set `DRAW_2D_BOXES=1` to additionally render the detector's 2D boxes.

## Outputs

- `rfdetr_bottom_envelope.jpg`: segmentation, bottom envelope, ranked contact
  points, ego-nearest contact, occlusion label, and either visible 3D faces
  or a segmentation-aligned polygon.
- `rfdetr_ground_contacts.json`: one record per detection.  Important fields:
  `ground_contacts`, `ground_contacts_global`,
  `ego_nearest_ground_contact`, `occlusion`, `estimated_visible_face`, and
  `segmentation_aligned_polygon`.

`occlusion.skip_3d_face` is the authoritative flag for whether a visible 3D
face was omitted.  Box overlap alone is treated conservatively as
`possibly_occluded`; it prevents a fragile 3D fit but does not claim a
definitive physical occlusion.

## Limitations

This is a monocular, geometry-plus-prior estimate.  Hidden wheels, inaccurate
masks, non-flat roads, and an incorrect class prior can still make the metric
box unreliable.  The segmentation polygon fallback is deliberately preferred
over a forced 3D face whenever the visible evidence is incomplete.
