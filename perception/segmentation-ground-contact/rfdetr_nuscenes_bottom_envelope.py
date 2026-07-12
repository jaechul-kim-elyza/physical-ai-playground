import os
import json
import numpy as np
from PIL import Image

import cv2

from nuscenes.nuscenes import NuScenes

from rfdetr import RFDETRSegMedium

import supervision as sv


def bottom_envelope(mask: np.ndarray) -> np.ndarray:
    """Return the lowest mask pixel for every x coordinate.

    Image coordinates grow downwards, so the largest y value is the pixel
    closest to the ground in this image-based approximation.
    """
    points = []
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            points.append((x, int(ys[-1])))
    return np.asarray(points, dtype=np.int32)


def extract_ground_contacts(
    mask: np.ndarray,
    *,
    vertical_tolerance_ratio: float = 0.025,
    min_separation_ratio: float = 0.08,
) -> tuple[np.ndarray, list[dict]]:
    """Extract visible ground-contact candidates from one instance mask.

    The bottom envelope is thresholded to a narrow band above its lowest
    pixel.  Contiguous bands are usually the visible bottoms of wheels or
    feet.  One representative point is returned per band, which avoids
    treating every bottom-envelope pixel as a separate contact.

    This is a 2-D *visible-contact* estimate.  It cannot recover a wheel or
    foot hidden by occlusion, and assumes the camera is roughly upright.
    """
    envelope = bottom_envelope(mask)
    if envelope.size == 0:
        return envelope, []

    xs, ys = envelope[:, 0], envelope[:, 1]
    object_height = max(1, int(ys.max() - ys.min() + 1))
    tolerance = max(2, int(round(object_height * vertical_tolerance_ratio)))
    bottom_y = int(ys.max())

    # Only keep the part of the silhouette indistinguishable from its lowest
    # point.  With separated wheels this naturally produces separate runs.
    near_ground = ys >= bottom_y - tolerance
    candidate_indices = np.flatnonzero(near_ground)
    if candidate_indices.size == 0:
        return envelope, []

    # Split runs not only at missing x values but also when they are far apart
    # relative to the object's image width.  The latter suppresses duplicates
    # caused by small gaps/noise around a single tire.
    object_width = max(1, int(xs.max() - xs.min() + 1))
    max_gap = max(1, int(round(object_width * min_separation_ratio)))
    runs = []
    start = 0
    for i in range(1, candidate_indices.size):
        previous_x = xs[candidate_indices[i - 1]]
        current_x = xs[candidate_indices[i]]
        if current_x - previous_x > max_gap:
            runs.append(candidate_indices[start:i])
            start = i
    runs.append(candidate_indices[start:])

    contacts = []
    for run in runs:
        run_y = ys[run]
        run_x = xs[run]
        lowest_y = int(run_y.max())
        # A flat wheel/foot bottom has several equal lowest pixels.  Its median
        # gives a stable point instead of choosing an arbitrary left edge.
        lowest_x = run_x[run_y == lowest_y]
        contact_x = int(np.median(lowest_x))
        support_width = int(run_x.max() - run_x.min() + 1)
        vertical_score = 1.0 - (bottom_y - lowest_y) / max(tolerance, 1)
        support_score = min(1.0, support_width / max(3, object_width * 0.08))
        contacts.append({
            "x": contact_x,
            "y": lowest_y,
            "score": round(float(0.8 * vertical_score + 0.2 * support_score), 3),
            "support_width_px": support_width,
        })

    # A side-view vehicle can have a front wheel that is visibly higher than
    # its rear wheel because of perspective.  It would be excluded by the
    # global lowest-point band above, so also collect pronounced *local*
    # bottoms of the envelope.  The smoothing prevents one-pixel mask noise
    # from becoming a contact point.
    smoothing_sigma = max(1.0, object_width * 0.01)
    smooth_y = cv2.GaussianBlur(
        ys.astype(np.float32).reshape(1, -1),
        (0, 0),
        sigmaX=smoothing_sigma,
    ).ravel()
    local_radius = max(3, int(round(object_width * 0.05)))
    local_kernel = np.ones((1, 2 * local_radius + 1), dtype=np.uint8)
    local_max = cv2.dilate(smooth_y.reshape(1, -1), local_kernel).ravel()
    lower_envelope_limit = np.percentile(ys, 65)
    local_indices = np.flatnonzero(
        (smooth_y >= local_max - 0.25) & (ys >= lower_envelope_limit)
    )

    if local_indices.size:
        local_runs = []
        start = 0
        for i in range(1, local_indices.size):
            if local_indices[i] != local_indices[i - 1] + 1:
                local_runs.append(local_indices[start:i])
                start = i
        local_runs.append(local_indices[start:])

        for run in local_runs:
            run_y = ys[run]
            run_x = xs[run]
            lowest_y = int(run_y.max())
            lowest_x = run_x[run_y == lowest_y]
            contact_x = int(np.median(lowest_x))
            support_width = int(run_x.max() - run_x.min() + 1)

            # Discard a nearly flat lower body edge.  A wheel/foot bottom is
            # lower than its neighbouring envelope by at least a few pixels.
            center = int(np.median(run))
            left = max(0, center - local_radius)
            right = min(len(ys) - 1, center + local_radius)
            neighbour_y = (smooth_y[left] + smooth_y[right]) / 2
            prominence = smooth_y[center] - neighbour_y
            if prominence < max(2.0, tolerance * 0.5):
                continue

            local_score = 0.6 + 0.4 * (
                (lowest_y - lower_envelope_limit)
                / max(1.0, bottom_y - lower_envelope_limit)
            )
            candidate = {
                "x": contact_x,
                "y": lowest_y,
                "score": round(float(np.clip(local_score, 0.0, 1.0)), 3),
                "support_width_px": support_width,
            }

            # A global and a local method can choose the same wheel.  Keep
            # one candidate, preferring the higher confidence measurement.
            nearest = next(
                (
                    existing for existing in contacts
                    if abs(existing["x"] - contact_x) <= local_radius
                ),
                None,
            )
            if nearest is None:
                contacts.append(candidate)
            elif candidate["score"] > nearest["score"]:
                contacts[contacts.index(nearest)] = candidate

    # Highest score first, then left-to-right for deterministic ties.
    contacts.sort(key=lambda p: (-p["score"], p["x"]))
    return envelope, contacts


# ==========================
# 1. nuScenes
# ==========================

DATA_ROOT = "/home/kim/datasets/nuScenes"
# Example side-view SUV (front and rear wheels are clearly visible):
# NUSCENES_CAMERA_CHANNEL=CAM_BACK_LEFT \
# NUSCENES_SAMPLE_TOKEN=d7387fb5a21d40a990a5842cca61af1c \
# .venv/bin/python perception/segmentation-ground-contact/rfdetr_nuscenes_bottom_envelope.py
CAMERA_CHANNEL = os.getenv("NUSCENES_CAMERA_CHANNEL", "CAM_FRONT")
SAMPLE_TOKEN = os.getenv("NUSCENES_SAMPLE_TOKEN")


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATA_ROOT,
    verbose=True
)


# ==========================
# 2. Load CAM_FRONT image
# ==========================

if SAMPLE_TOKEN:
    sample = nusc.get("sample", SAMPLE_TOKEN)
else:
    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])


if CAMERA_CHANNEL not in sample["data"]:
    raise ValueError(
        f"{CAMERA_CHANNEL} is not available for sample {sample['token']}"
    )

cam_token = sample["data"][CAMERA_CHANNEL]


cam_data = nusc.get(
    "sample_data",
    cam_token
)


img_path = os.path.join(
    DATA_ROOT,
    cam_data["filename"]
)


print("Image:")
print(img_path)
print("Camera channel:", CAMERA_CHANNEL)


image = Image.open(img_path).convert("RGB")

print("Image size:", image.size)



# ==========================
# 3. RF-DETR Seg
# ==========================

print("Loading RF-DETR...")


model = RFDETRSegMedium()



# ==========================
# 4. Inference
# ==========================

print("Inference...")


detections = model.predict(
    image,
    threshold=0.5
)


print(type(detections))

print(detections)



# ==========================
# 5. Visualization
# ==========================

print("Visualization...")


annotated = np.array(image.copy())


# mask
mask_annotator = sv.MaskAnnotator()

annotated = mask_annotator.annotate(
    scene=annotated,
    detections=detections
)


# bbox
box_annotator = sv.BoxAnnotator()

annotated = box_annotator.annotate(
    scene=annotated,
    detections=detections
)


# label
label_annotator = sv.LabelAnnotator()

annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections
)



# ==========================
# 6. Bottom envelope extraction
# ==========================

print("Extracting bottom pixels...")


ground_contact_results = []

if detections.mask is not None:

    masks = detections.mask


    for obj_id, mask in enumerate(masks):
        bottom_points, contacts = extract_ground_contacts(mask)

        class_id = (
            int(detections.class_id[obj_id])
            if detections.class_id is not None else None
        )
        confidence = (
            float(detections.confidence[obj_id])
            if detections.confidence is not None else None
        )
        result = {
            "object_id": obj_id,
            "class_id": class_id,
            "detection_confidence": confidence,
            # Ordered by estimated contact likelihood (score), not x position.
            "ground_contacts": contacts,
        }
        ground_contact_results.append(result)


        print(
            f"Object {obj_id}: {len(contacts)} contact candidate(s)",
            contacts
        )


        # ==========================
        # draw bottom envelope
        # ==========================

        for x, y in bottom_points:

            cv2.circle(
                annotated,
                (int(x), int(y)),
                2,
                (255, 0, 0),
                -1
            )

        # Yellow cross: selected visible ground-contact point.  The numeric
        # score is a relative ranking within this object's bottom silhouette.
        for rank, contact in enumerate(contacts, start=1):
            point = (contact["x"], contact["y"])
            cv2.drawMarker(
                annotated, point, (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )
            cv2.putText(
                annotated,
                str(rank),
                (point[0] + 4, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )


else:

    print("No segmentation mask")



# ==========================
# 7. Save
# ==========================

out_file = "rfdetr_bottom_envelope.jpg"
contacts_file = "rfdetr_ground_contacts.json"


Image.fromarray(
    annotated
).save(out_file)


print("====================")
print("Saved:")
print(out_file)
with open(contacts_file, "w", encoding="utf-8") as f:
    json.dump(ground_contact_results, f, ensure_ascii=False, indent=2)

print(contacts_file)
print("====================")
