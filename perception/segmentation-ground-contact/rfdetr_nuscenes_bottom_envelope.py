import os
import json
import numpy as np
from PIL import Image

import cv2

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from rfdetr import RFDETRSegMedium

import supervision as sv


# Dimensions are priors in metres, not measurements from the image.  They
# resolve the scale ambiguity of a single camera after contact points anchor
# the object to the ground plane.
VEHICLE_SIZE_PRIORS_LWH = {
    "car": (4.5, 1.8, 1.6),
    "truck": (7.0, 2.5, 3.0),
    "bus": (11.0, 2.6, 3.2),
    "motorcycle": (2.1, 0.8, 1.2),
    "bicycle": (1.8, 0.6, 1.2),
}

def camera_to_global_transform(nusc: NuScenes, cam_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation and origin that transform a camera point into global coordinates."""
    calibrated_sensor = nusc.get(
        "calibrated_sensor", cam_data["calibrated_sensor_token"]
    )
    ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])

    rotation_ego_camera = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
    translation_ego_camera = np.asarray(calibrated_sensor["translation"], dtype=float)
    rotation_global_ego = Quaternion(ego_pose["rotation"]).rotation_matrix
    translation_global_ego = np.asarray(ego_pose["translation"], dtype=float)

    rotation_global_camera = rotation_global_ego @ rotation_ego_camera
    origin_global_camera = (
        rotation_global_ego @ translation_ego_camera + translation_global_ego
    )
    return rotation_global_camera, origin_global_camera


def contacts_to_ground_global(
    contacts: list[dict],
    camera_intrinsic: np.ndarray,
    rotation_global_camera: np.ndarray,
    origin_global_camera: np.ndarray,
    ground_z: float,
) -> list[dict]:
    """Intersect contact-pixel rays with a horizontal ground plane in global frame."""
    inverse_intrinsic = np.linalg.inv(camera_intrinsic)
    points_global = []

    for contact in contacts:
        ray_camera = inverse_intrinsic @ np.array(
            [contact["x"], contact["y"], 1.0], dtype=float
        )
        ray_global = rotation_global_camera @ ray_camera
        if abs(ray_global[2]) < 1e-8:
            continue

        distance = (ground_z - origin_global_camera[2]) / ray_global[2]
        if distance <= 0:
            continue

        point = origin_global_camera + distance * ray_global
        points_global.append({
            "x": round(float(point[0]), 3),
            "y": round(float(point[1]), 3),
            "z": round(float(point[2]), 3),
            "source_pixel": [contact["x"], contact["y"]],
            "score": contact["score"],
        })
    return points_global


def mark_ego_nearest_contact(
    contacts_global: list[dict], ego_origin_global: np.ndarray
) -> dict | None:
    """Mark and return the visible ground contact nearest to the ego vehicle."""
    if not contacts_global:
        return None

    ego_xy = np.asarray(ego_origin_global[:2], dtype=float)
    for contact in contacts_global:
        contact_xy = np.array([contact["x"], contact["y"]], dtype=float)
        contact["distance_to_ego_m"] = round(
            float(np.linalg.norm(contact_xy - ego_xy)), 3
        )

    nearest_index = min(
        range(len(contacts_global)),
        key=lambda index: contacts_global[index]["distance_to_ego_m"],
    )
    for index, contact in enumerate(contacts_global):
        contact["is_nearest_to_ego"] = index == nearest_index
    return contacts_global[nearest_index]


def estimate_vehicle_3d_box(
    contacts_global: list[dict],
    class_name: str | None,
    ground_z: float,
    ego_origin_global: np.ndarray,
) -> dict | None:
    """Estimate a vehicle box from visible wheel contacts and a size prior.

    The widest pair of ground contacts is treated as the front/rear wheel
    axis.  In a side view this wheel line belongs to the vehicle side visible
    to the ego camera, not to its lateral centre line.  The box is therefore
    shifted by half its width *away* from ego, anchoring its closest side to
    the observed contact line.  Metric dimensions remain class priors because
    one image cannot independently determine a vehicle's true scale.
    """
    if class_name not in VEHICLE_SIZE_PRIORS_LWH or len(contacts_global) < 2:
        return None

    xy = np.asarray([[point["x"], point["y"]] for point in contacts_global])
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    wheel_axis = xy[second] - xy[first]
    wheelbase_observed = float(np.linalg.norm(wheel_axis))
    if wheelbase_observed < 0.2:
        return None

    length, width, height = VEHICLE_SIZE_PRIORS_LWH[class_name]
    visible_wheel_line_midpoint = (xy[first] + xy[second]) / 2
    axis_unit = wheel_axis / wheelbase_observed
    # Remove the length-axis component: this leaves the ego direction that is
    # perpendicular to the observed wheel line.  It identifies which side of
    # the vehicle is closest to the ego vehicle.
    ego_vector = np.asarray(ego_origin_global[:2]) - visible_wheel_line_midpoint
    ego_side_vector = ego_vector - np.dot(ego_vector, axis_unit) * axis_unit
    ego_side_distance = float(np.linalg.norm(ego_side_vector))
    if ego_side_distance < 0.05:
        # Degenerate frontal/rear view: preserve the old centre-line estimate
        # rather than applying an arbitrary lateral shift.
        ground_center = visible_wheel_line_midpoint
        side_anchor_used = False
    else:
        direction_toward_ego = ego_side_vector / ego_side_distance
        ground_center = visible_wheel_line_midpoint - direction_toward_ego * width / 2
        side_anchor_used = True
    yaw = float(np.arctan2(wheel_axis[1], wheel_axis[0]))
    return {
        "frame": "global",
        "center_xyz": [
            round(float(ground_center[0]), 3),
            round(float(ground_center[1]), 3),
            round(float(ground_z + height / 2), 3),
        ],
        "size_lwh_m": [length, width, height],
        "yaw_rad": round(yaw, 4),
        "wheelbase_observed_m": round(wheelbase_observed, 3),
        "dimension_source": "class_prior",
        "side_anchor": (
            "visible_wheel_line_shifted_away_from_ego"
            if side_anchor_used else "center_line_fallback"
        ),
        "visible_wheel_line_midpoint_xy": [
            round(float(visible_wheel_line_midpoint[0]), 3),
            round(float(visible_wheel_line_midpoint[1]), 3),
        ],
    }


def global_box_corners(box: dict) -> np.ndarray:
    """Return the eight 3D box corners in global coordinates."""
    length, width, height = box["size_lwh_m"]
    center = np.asarray(box["center_xyz"], dtype=float)
    yaw = box["yaw_rad"]

    # First four corners are on the ground; the last four are their top-face
    # counterparts.  The vehicle length axis follows its estimated yaw.
    local_corners = np.array([
        [length / 2, width / 2, -height / 2],
        [length / 2, -width / 2, -height / 2],
        [-length / 2, -width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
        [length / 2, width / 2, height / 2],
        [length / 2, -width / 2, height / 2],
        [-length / 2, -width / 2, height / 2],
        [-length / 2, width / 2, height / 2],
    ])
    rotation_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return (rotation_z @ local_corners.T).T + center


def project_3d_box_to_image(
    box: dict,
    camera_intrinsic: np.ndarray,
    rotation_global_camera: np.ndarray,
    origin_global_camera: np.ndarray,
) -> np.ndarray | None:
    """Project global-frame 3D box corners into the current camera image."""
    corners_global = global_box_corners(box)

    # rotation_global_camera transforms camera -> global, so transpose it for
    # global -> camera.  A box behind the camera must not be drawn.
    corners_camera = (
        rotation_global_camera.T @ (corners_global - origin_global_camera).T
    ).T
    if np.any(corners_camera[:, 2] <= 0.1):
        return None

    projected = (camera_intrinsic @ corners_camera.T).T
    return projected[:, :2] / projected[:, 2:3]


def visible_vehicle_face(box: dict, ego_origin_global: np.ndarray) -> dict:
    """Return the visible vertical vehicle face(s) from the ego viewpoint.

    A strictly side-on view still has a small front/rear component in a
    perspective image.  When that component is material, return both adjacent
    faces rather than dropping the vehicle's visible end cap.
    """
    yaw = box["yaw_rad"]
    center_xy = np.asarray(box["center_xyz"][:2], dtype=float)
    heading_axis = np.array([np.cos(yaw), np.sin(yaw)])
    # Positive local-width is vehicle-left for a length axis pointing at yaw.
    vehicle_left_axis = np.array([-np.sin(yaw), np.cos(yaw)])
    view_vector = np.asarray(ego_origin_global[:2]) - center_xy
    view_vector /= max(np.linalg.norm(view_vector), 1e-8)
    longitudinal = float(np.dot(view_vector, heading_axis))
    lateral = float(np.dot(view_vector, vehicle_left_axis))
    view_angle_deg = float(np.degrees(np.arctan2(lateral, longitudinal)))

    longitudinal_face = "front" if longitudinal >= 0 else "rear"
    lateral_face = "left" if lateral >= 0 else "right"
    # A 30-degree cone yields an unambiguous face.  Between the two cones the
    # camera sees a corner; retain that fact while drawing its dominant face.
    if abs(longitudinal) >= abs(lateral) * np.sqrt(3):
        visible_surface = longitudinal_face
        primary_face = longitudinal_face
    elif abs(lateral) >= abs(longitudinal) * np.sqrt(3):
        visible_surface = lateral_face
        primary_face = lateral_face
    else:
        visible_surface = f"{longitudinal_face}_{lateral_face}_oblique"
        primary_face = (
            longitudinal_face
            if abs(longitudinal) >= abs(lateral) else lateral_face
        )

    face_indices = {
        "front": (0, 1, 5, 4),
        "rear": (3, 2, 6, 7),
        "left": (0, 3, 7, 4),
        "right": (1, 2, 6, 5),
    }
    corners = global_box_corners(box)
    secondary_face = lateral_face if primary_face == longitudinal_face else longitudinal_face
    primary_component = max(abs(longitudinal), abs(lateral))
    secondary_component = min(abs(longitudinal), abs(lateral))
    # About seven degrees away from a pure face-on/side-on view.  This keeps
    # a genuinely edge-on end cap out of the result but covers the rear/front
    # visible in examples such as sample_10.
    visible_faces = [primary_face]
    if secondary_component / max(primary_component, 1e-8) >= 0.12:
        visible_faces.append(secondary_face)
    indices = face_indices[primary_face]

    face_parts = []
    for name in visible_faces:
        part_indices = face_indices[name]
        face_parts.append({
            "name": name,
            "corner_indices": list(part_indices),
            "corners_xyz": [
                [round(float(value), 3) for value in corner]
                for corner in corners[list(part_indices)]
            ],
        })
    return {
        "frame": "global",
        "visible_surface": visible_surface,
        "primary_face": primary_face,
        "visible_faces": face_parts,
        "view_angle_deg": round(view_angle_deg, 2),
        # Retained for consumers of the earlier single-face JSON schema.
        "corner_indices": list(indices),
        "corners_xyz": [
            [round(float(value), 3) for value in corner]
            for corner in corners[list(indices)]
        ],
        "segmentation_fit": box.get("segmentation_fit"),
        "segmentation_refinement": box.get("visible_face_refinement"),
    }


def box_with_dimensions_anchored_to_near_side(
    box: dict,
    dimensions_lwh: np.ndarray,
    ego_origin_global: np.ndarray,
    longitudinal_offset_m: float = 0.0,
) -> dict:
    """Resize a box while preserving its visible wheel-line side anchor."""
    length, width, height = [float(value) for value in dimensions_lwh]
    resized = dict(box)
    resized["size_lwh_m"] = [length, width, height]

    ground_z = box["center_xyz"][2] - box["size_lwh_m"][2] / 2
    midpoint = np.asarray(box["visible_wheel_line_midpoint_xy"], dtype=float)
    axis = np.array([np.cos(box["yaw_rad"]), np.sin(box["yaw_rad"])])
    ego_vector = np.asarray(ego_origin_global[:2]) - midpoint
    ego_side = ego_vector - np.dot(ego_vector, axis) * axis
    ego_side_norm = np.linalg.norm(ego_side)

    if box["side_anchor"].startswith("visible_wheel_line") and ego_side_norm >= 0.05:
        center_xy = midpoint - ego_side / ego_side_norm * width / 2
    else:
        center_xy = midpoint
    center_xy = center_xy + axis * longitudinal_offset_m
    resized["center_xyz"] = [
        round(float(center_xy[0]), 6),
        round(float(center_xy[1]), 6),
        round(float(ground_z + height / 2), 6),
    ]
    resized["segmentation_longitudinal_offset_m"] = round(
        float(longitudinal_offset_m), 3
    )
    return resized


def segmentation_coverage_of_projected_box(
    mask_contour: np.ndarray,
    projected_corners: np.ndarray | None,
) -> float:
    """Fraction of segmentation contour enclosed by projected 3D-box bounds."""
    if projected_corners is None or mask_contour.size == 0:
        return 0.0
    points = mask_contour.reshape(-1, 2)
    # A few hundred contour points retain the complete silhouette while making
    # the dimension search inexpensive for large masks.
    stride = max(1, len(points) // 600)
    points = points[::stride]
    minimum = projected_corners.min(axis=0)
    maximum = projected_corners.max(axis=0)
    inside = np.logical_and.reduce((
        points[:, 0] >= minimum[0],
        points[:, 0] <= maximum[0],
        points[:, 1] >= minimum[1],
        points[:, 1] <= maximum[1],
    )).sum()
    return inside / len(points)


def fit_3d_box_to_segmentation(
    box: dict | None,
    mask: np.ndarray,
    camera_intrinsic: np.ndarray,
    rotation_global_camera: np.ndarray,
    origin_global_camera: np.ndarray,
    ego_origin_global: np.ndarray,
) -> dict | None:
    """Expand/shrink a 3D box until its projection tightly contains the mask.

    The ground-contact pose stays fixed.  Only L/W/H are optimized, with the
    near-side wheel-line anchor recomputed when the width changes.
    """
    if box is None:
        return None

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return box
    # RF-DETR masks can contain tiny disconnected speckles.  Fit the physical
    # object component, not an isolated prediction artifact metres away from
    # the vehicle silhouette.
    contour = max(contours, key=cv2.contourArea)

    def evaluate(candidate: dict) -> float:
        return segmentation_coverage_of_projected_box(
            contour,
            project_3d_box_to_image(
                candidate,
                camera_intrinsic,
                rotation_global_camera,
                origin_global_camera,
            ),
        )

    dimensions = np.asarray(box["size_lwh_m"], dtype=float)
    candidate = box_with_dimensions_anchored_to_near_side(
        box, dimensions, ego_origin_global
    )
    coverage = evaluate(candidate)

    # Search a compact L/W/H grid rather than greedily expanding one axis.
    # Width changes the ego-side anchor, so greedy growth can occasionally
    # make a projection worse even though the physical box gets larger.
    if coverage < 0.999:
        # If wheel contacts are detected on only one end of a vehicle, the
        # initial wheel-line pose can put most of the silhouette outside the
        # class-prior projection.  Use a wider recovery search only for that
        # case; normal fits retain the compact, physically plausible search.
        severe_undercoverage = coverage < 0.70
        length_scales = (
            (1.0, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
            if severe_undercoverage else
            (1.0, 1.05, 1.10, 1.20, 1.35, 1.50)
        )
        width_scales = (1.0, 1.10, 1.25)
        height_scales = (
            (1.0, 1.15, 1.35, 1.60, 2.00, 2.50, 3.00)
            if severe_undercoverage else
            (1.0, 1.05, 1.10, 1.20, 1.35, 1.50)
        )
        longitudinal_offsets = (
            (-3.0, -2.0, -1.5, -1.0, -0.6, -0.3, 0.0,
             0.3, 0.6, 1.0, 1.5, 2.0, 3.0)
            if severe_undercoverage else
            (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)
        )
        containing_options = []
        best_option = (coverage, dimensions, candidate, 0.0)
        for length_scale in length_scales:
            for width_scale in width_scales:
                for height_scale in height_scales:
                    for offset in longitudinal_offsets:
                        scaled_dimensions = dimensions * np.array([
                            length_scale, width_scale, height_scale
                        ])
                        scaled_box = box_with_dimensions_anchored_to_near_side(
                            box, scaled_dimensions, ego_origin_global, offset
                        )
                        scaled_coverage = evaluate(scaled_box)
                        option = (scaled_coverage, scaled_dimensions, scaled_box, offset)
                        if scaled_coverage >= 0.999:
                            containing_options.append(option)
                        if scaled_coverage > best_option[0]:
                            best_option = option

        if containing_options:
            coverage, dimensions, candidate, longitudinal_offset = min(
                containing_options, key=lambda option: float(np.prod(option[1]))
            )
        else:
            coverage, dimensions, candidate, longitudinal_offset = best_option
    else:
        longitudinal_offset = 0.0

    # Then contract each dimension wherever coverage remains complete.  This
    # removes spare prior-size padding and gives a tight enclosing projection.
    for _ in range(30):
        changed = False
        # A side camera provides little evidence for the hidden vehicle width;
        # retain its class prior instead of shrinking it to fit a 2-D outline.
        for dimension_index in (0, 2):
            shrunk = dimensions.copy()
            shrunk[dimension_index] *= 0.98
            shrunk_box = box_with_dimensions_anchored_to_near_side(
                box, shrunk, ego_origin_global, longitudinal_offset
            )
            shrunk_coverage = evaluate(shrunk_box)
            if shrunk_coverage >= 0.999:
                dimensions, candidate, coverage = shrunk, shrunk_box, shrunk_coverage
                changed = True
        if not changed:
            break

    candidate["size_lwh_m"] = [round(float(value), 3) for value in dimensions]
    candidate["segmentation_fit"] = {
        "method": "projected_3d_bounds_enclose_mask",
        "contour_coverage": round(float(coverage), 4),
    }
    return candidate


def draw_3d_box(
    image: np.ndarray,
    projected_corners: np.ndarray | None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw a projected 3D box as a wireframe directly onto an RGB image."""
    if projected_corners is None:
        return

    corners = np.rint(projected_corners).astype(np.int32)
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),  # ground face
        (4, 5), (5, 6), (6, 7), (7, 4),  # roof face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    )
    for start, end in edges:
        cv2.line(image, tuple(corners[start]), tuple(corners[end]), color, 3)


def draw_visible_vehicle_face(
    image: np.ndarray,
    projected_corners: np.ndarray | None,
    face: dict | None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw every materially visible vertical face of the vehicle."""
    if projected_corners is None or face is None:
        return

    overlay = image.copy()
    parts = face.get("visible_faces", [{
        "name": face["primary_face"],
        "corner_indices": face["corner_indices"],
    }])
    projected_parts = []
    for part in parts:
        corners = np.rint(
            projected_corners[part["corner_indices"]]
        ).astype(np.int32)
        cv2.fillConvexPoly(overlay, corners, color)
        projected_parts.append(corners)
    cv2.addWeighted(overlay, 0.18, image, 0.82, 0, dst=image)
    for corners in projected_parts:
        cv2.polylines(image, [corners], isClosed=True, color=color, thickness=3)
    label_point = tuple(
        np.vstack(projected_parts).mean(axis=0).astype(np.int32)
    )
    label = " + ".join(part["name"].upper() for part in parts)
    cv2.putText(
        image,
        label,
        (label_point[0] - 45, label_point[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def visible_face_mask_metrics(
    mask: np.ndarray,
    projected_corners: np.ndarray | None,
    face: dict | None,
) -> tuple[float, float]:
    """Return IoU and mask coverage for one projected visible face."""
    face_mask = projected_visible_face_mask(mask.shape, projected_corners, face)
    if face_mask is None:
        return 0.0, 0.0
    object_mask = mask.astype(bool)
    intersection = np.logical_and(object_mask, face_mask).sum()
    union = np.logical_or(object_mask, face_mask).sum()
    iou = float(intersection / union) if union else 0.0
    coverage = float(intersection / object_mask.sum()) if object_mask.any() else 0.0
    return iou, coverage


def projected_visible_face_mask(
    image_shape: tuple[int, int],
    projected_corners: np.ndarray | None,
    face: dict | None,
) -> np.ndarray | None:
    """Rasterize a selected 3D face in image coordinates."""
    if projected_corners is None or face is None:
        return None
    face_mask = np.zeros(image_shape, dtype=np.uint8)
    parts = face.get("visible_faces", [{
        "corner_indices": face["corner_indices"],
    }])
    for part in parts:
        vertices = np.rint(
            projected_corners[part["corner_indices"]]
        ).astype(np.int32)
        cv2.fillConvexPoly(face_mask, vertices, 1)
    return face_mask.astype(bool)


def infer_occlusion(
    mask: np.ndarray,
    projected_corners: np.ndarray | None,
    face: dict | None,
    all_masks: np.ndarray,
    detection_boxes: np.ndarray,
    object_id: int,
) -> dict:
    """Record conservative occlusion evidence from masks and detector boxes."""
    object_mask = mask.astype(bool)
    touches_image_edge = bool(
        object_mask[0].any() or object_mask[-1].any()
        or object_mask[:, 0].any() or object_mask[:, -1].any()
    )
    face_mask = projected_visible_face_mask(mask.shape, projected_corners, face)
    box = detection_boxes[object_id]
    bbox_overlaps = []
    for other_id, other_box in enumerate(detection_boxes):
        if other_id == object_id:
            continue
        left = max(float(box[0]), float(other_box[0]))
        top = max(float(box[1]), float(other_box[1]))
        right = min(float(box[2]), float(other_box[2]))
        bottom = min(float(box[3]), float(other_box[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        if intersection <= 0:
            continue
        box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        other_area = max(0.0, float(other_box[2] - other_box[0])) * max(0.0, float(other_box[3] - other_box[1]))
        iou = intersection / max(box_area + other_area - intersection, 1e-8)
        smaller_box_fraction = intersection / max(min(box_area, other_area), 1e-8)
        bbox_overlaps.append({
            "object_id": other_id,
            "iou": round(float(iou), 4),
            "smaller_box_overlap_fraction": round(float(smaller_box_fraction), 4),
        })
    max_bbox_iou = max((entry["iou"] for entry in bbox_overlaps), default=0.0)
    max_bbox_fraction = max(
        (entry["smaller_box_overlap_fraction"] for entry in bbox_overlaps),
        default=0.0,
    )

    if face_mask is None:
        return {
            "status": (
                "truncated" if touches_image_edge
                else "possibly_occluded"
                if max_bbox_iou >= 0.05 or max_bbox_fraction >= 0.10
                else "unknown"
            ),
            "reason": "no_metric_visible_face",
            "bbox_overlap_iou": max_bbox_iou,
            "bbox_smaller_box_overlap_fraction": max_bbox_fraction,
            "bbox_overlapping_object_ids": [
                entry["object_id"] for entry in bbox_overlaps
            ],
            "touches_image_edge": touches_image_edge,
        }

    expected_area = max(1, int(face_mask.sum()))
    visible_coverage = float(np.logical_and(face_mask, object_mask).sum() / expected_area)
    missing_face = np.logical_and(face_mask, ~object_mask)
    occluder_ids = []
    occluder_pixels = 0
    for other_id, other_mask in enumerate(all_masks):
        if other_id == object_id:
            continue
        overlap = int(np.logical_and(missing_face, other_mask.astype(bool)).sum())
        if overlap:
            occluder_ids.append(other_id)
            occluder_pixels += overlap
    occluder_ratio = occluder_pixels / expected_area

    if touches_image_edge:
        status = "truncated"
    elif visible_coverage < 0.90 and occluder_ratio >= 0.03:
        status = "occluded"
    elif max_bbox_iou >= 0.05 or max_bbox_fraction >= 0.10:
        # A detector-box overlap is useful evidence, but not definitive: two
        # nearby objects can overlap in 2D without one hiding the other.
        status = "possibly_occluded"
    elif visible_coverage >= 0.90:
        status = "not_occluded"
    else:
        # A vehicle silhouette has wheel arches/windows that also lower face
        # coverage.  Without another mask in the missing region this is not
        # enough evidence to call it occlusion.
        status = "unknown"
    return {
        "status": status,
        "reason": "visible_face_mask_comparison",
        "visible_face_coverage": round(visible_coverage, 4),
        "other_mask_occluder_ratio": round(occluder_ratio, 4),
        "occluder_object_ids": occluder_ids,
        "bbox_overlap_iou": max_bbox_iou,
        "bbox_smaller_box_overlap_fraction": max_bbox_fraction,
        "bbox_overlapping_object_ids": [
            entry["object_id"] for entry in bbox_overlaps
        ],
        "touches_image_edge": touches_image_edge,
    }


def draw_occlusion_status(
    image: np.ndarray,
    mask: np.ndarray,
    occlusion: dict,
) -> None:
    """Render the conservative occlusion assessment next to one object mask."""
    ys, xs = np.where(mask)
    if not xs.size:
        return
    status = occlusion["status"]
    labels = {
        "occluded": "OCCLUDED",
        "possibly_occluded": "POSSIBLE OCC",
        "truncated": "TRUNCATED",
        "not_occluded": "CLEAR",
        "unknown": "UNKNOWN",
    }
    colors = {
        "occluded": (255, 0, 0),
        "possibly_occluded": (255, 165, 0),
        "truncated": (255, 165, 0),
        "not_occluded": (0, 255, 0),
        "unknown": (255, 255, 0),
    }
    text = f"OCC: {labels[status]}"
    position = (int(xs.min()), max(18, int(ys.min()) - 8))
    # A dark outline keeps the status legible on bright vehicles and road.
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        0.45, colors[status], 1, cv2.LINE_AA,
    )


def refine_visible_face_with_segmentation(
    box: dict | None,
    mask: np.ndarray,
    camera_intrinsic: np.ndarray,
    rotation_global_camera: np.ndarray,
    origin_global_camera: np.ndarray,
    ego_origin_global: np.ndarray,
) -> tuple[dict | None, dict | None]:
    """Accept a local face fit only when it improves mask overlap safely."""
    if box is None:
        return None, None

    def evaluate(candidate: dict) -> tuple[float, float, dict]:
        projected = project_3d_box_to_image(
            candidate,
            camera_intrinsic,
            rotation_global_camera,
            origin_global_camera,
        )
        face = visible_vehicle_face(candidate, ego_origin_global)
        iou, coverage = visible_face_mask_metrics(mask, projected, face)
        return iou, coverage, face

    baseline_iou, baseline_coverage, _ = evaluate(box)
    best_box = box
    best_iou = baseline_iou
    best_coverage = baseline_coverage
    base_dimensions = np.asarray(box["size_lwh_m"], dtype=float)
    base_offset = float(box.get("segmentation_longitudinal_offset_m", 0.0))
    minimum_coverage = max(0.90, baseline_coverage - 0.02)

    # Keep the ego-side anchor and class width prior.  Optimize only the two
    # dimensions visible in a side/front mask plus a small longitudinal shift.
    for length_scale in (0.88, 0.94, 1.0, 1.06, 1.12):
        for height_scale in (0.88, 0.94, 1.0, 1.06, 1.12):
            for offset_delta in (-0.4, -0.2, 0.0, 0.2, 0.4):
                dimensions = base_dimensions * np.array([
                    length_scale, 1.0, height_scale
                ])
                candidate = box_with_dimensions_anchored_to_near_side(
                    box,
                    dimensions,
                    ego_origin_global,
                    base_offset + offset_delta,
                )
                candidate_iou, candidate_coverage, _ = evaluate(candidate)
                if (
                    candidate_coverage >= minimum_coverage
                    and candidate_iou > best_iou + 0.005
                ):
                    best_box = candidate
                    best_iou = candidate_iou
                    best_coverage = candidate_coverage

    accepted = best_box is not box
    selected_box = best_box if accepted else box
    selected_box["visible_face_refinement"] = {
        "accepted": accepted,
        "baseline_mask_iou": round(float(baseline_iou), 4),
        "mask_iou": round(float(best_iou), 4),
        "mask_coverage": round(float(best_coverage), 4),
    }
    return selected_box, selected_box["visible_face_refinement"]


def segmentation_aligned_polygon(mask: np.ndarray) -> dict | None:
    """Return the actual exterior polygon of one segmentation component."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    return {
        "frame": "image",
        "vertices_xy": [
            [round(float(x), 1), round(float(y), 1)]
            for x, y in contour.reshape(-1, 2)
        ],
        "contour_coverage": 1.0,
        "method": "segmentation_exterior_contour",
    }


def draw_segmentation_polygon(
    image: np.ndarray,
    polygon: dict | None,
    color: tuple[int, int, int] = (255, 0, 255),
) -> None:
    """Draw the segmentation-aligned shape used when only one contact exists."""
    if polygon is None:
        return
    vertices = np.rint(polygon["vertices_xy"]).astype(np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [vertices], color)
    cv2.addWeighted(overlay, 0.14, image, 0.86, 0, dst=image)
    cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=3)
    label_point = tuple(vertices.mean(axis=0).astype(np.int32))
    cv2.putText(
        image,
        "MASK SHAPE",
        (label_point[0] - 40, label_point[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
        cv2.LINE_AA,
    )


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
DRAW_2D_BOXES = os.getenv("DRAW_2D_BOXES", "0") == "1"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".")


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

calibrated_sensor = nusc.get(
    "calibrated_sensor", cam_data["calibrated_sensor_token"]
)
ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
ego_origin_global = np.asarray(ego_pose["translation"], dtype=float)
camera_intrinsic = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=float)
rotation_global_camera, origin_global_camera = camera_to_global_transform(
    nusc, cam_data
)
# In nuScenes the ego frame is referenced to the road plane.  An environment
# override is useful for a dataset whose map/global origin is elsewhere.
GROUND_Z = float(os.getenv("NUSCENES_GROUND_Z", ego_pose["translation"][2]))


img_path = os.path.join(
    DATA_ROOT,
    cam_data["filename"]
)


print("Image:")
print(img_path)
print("Camera channel:", CAMERA_CHANNEL)
print("Ground plane (global z):", GROUND_Z)


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

print("All-class detections:", len(detections))


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


# The green polygon below is the ego-facing estimated 3D face.  Keep 2D
# detection boxes optional so they do not obscure it.
if DRAW_2D_BOXES:
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
        class_name = (
            str(detections.data["class_name"][obj_id])
            if "class_name" in detections.data else None
        )
        contacts_global = contacts_to_ground_global(
            contacts,
            camera_intrinsic,
            rotation_global_camera,
            origin_global_camera,
            GROUND_Z,
        )
        ego_nearest_contact = mark_ego_nearest_contact(
            contacts_global, ego_origin_global
        )
        estimated_3d_box = estimate_vehicle_3d_box(
            contacts_global, class_name, GROUND_Z, ego_origin_global
        )
        estimated_3d_box = fit_3d_box_to_segmentation(
            estimated_3d_box,
            mask,
            camera_intrinsic,
            rotation_global_camera,
            origin_global_camera,
            ego_origin_global,
        )
        estimated_3d_box, _ = refine_visible_face_with_segmentation(
            estimated_3d_box,
            mask,
            camera_intrinsic,
            rotation_global_camera,
            origin_global_camera,
            ego_origin_global,
        )
        projected_3d_box = project_3d_box_to_image(
            estimated_3d_box,
            camera_intrinsic,
            rotation_global_camera,
            origin_global_camera,
        ) if estimated_3d_box is not None else None
        estimated_visible_face = (
            visible_vehicle_face(estimated_3d_box, ego_origin_global)
            if estimated_3d_box is not None else None
        )
        occlusion = infer_occlusion(
            mask,
            projected_3d_box,
            estimated_visible_face,
            masks,
            detections.xyxy,
            obj_id,
        )
        # A partial/overlapped silhouette does not constrain a physical face
        # reliably.  Keep its 2-D segmentation and contact evidence, but do
        # not publish or render a 3-D face for it.  ``possibly_occluded`` is
        # included deliberately: bbox overlap is enough to make the fitted
        # depth/yaw unstable, even if it is not definitive proof of hiding.
        skip_3d_face = occlusion["status"] in {
            "occluded",
            "possibly_occluded",
            "truncated",
        }
        occlusion["skip_3d_face"] = skip_3d_face
        if skip_3d_face:
            estimated_3d_box = None
            projected_3d_box = None
            estimated_visible_face = None
        segmentation_polygon = (
            segmentation_aligned_polygon(mask)
            # Any class without a metric 3D face still receives its exact
            # segmentation shape, including occluded vehicles, people and
            # non-road objects.
            if estimated_visible_face is None else None
        )
        result = {
            "object_id": obj_id,
            "class_id": class_id,
            "class_name": class_name,
            "detection_confidence": confidence,
            # Ordered by estimated contact likelihood (score), not x position.
            "ground_contacts": contacts,
            "ground_contacts_global": contacts_global,
            # Representative contact: physically closest visible ground point
            # to the ego vehicle, computed after 3D ground-plane projection.
            "ego_nearest_ground_contact": ego_nearest_contact,
            "occlusion": occlusion,
            # Only the ego-facing vehicle face is emitted.  The full cuboid is
            # an internal fitting aid and is intentionally not exported.
            "estimated_visible_face": estimated_visible_face,
            # A single contact cannot constrain yaw/depth reliably.  Return
            # the segmentation's own image-space exterior polygon instead.
            "segmentation_aligned_polygon": segmentation_polygon,
        }
        ground_contact_results.append(result)


        print(
            f"Object {obj_id}: {len(contacts)} contact candidate(s)",
            contacts
        )
        print("Occlusion:", occlusion)
        draw_occlusion_status(annotated, mask, occlusion)
        if estimated_3d_box is not None:
            print("Estimated visible 3D face:", estimated_visible_face)
            draw_visible_vehicle_face(
                annotated, projected_3d_box, estimated_visible_face
            )
        elif segmentation_polygon is not None:
            if skip_3d_face:
                print("Skipped visible 3D face due to occlusion evidence")
            print(
                "Segmentation-aligned polygon:",
                f"vertices={len(segmentation_polygon['vertices_xy'])}",
                f"coverage={segmentation_polygon['contour_coverage']}",
            )
            draw_segmentation_polygon(annotated, segmentation_polygon)
        if ego_nearest_contact is not None:
            print("Ego-nearest ground contact:", ego_nearest_contact)


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

        # Green diamond: the candidate that is closest to the ego vehicle on
        # the reconstructed ground plane.  This is the representative contact
        # used when a downstream consumer needs a single point per object.
        if ego_nearest_contact is not None:
            nearest_point = tuple(ego_nearest_contact["source_pixel"])
            cv2.drawMarker(
                annotated,
                nearest_point,
                (0, 255, 0),
                markerType=cv2.MARKER_DIAMOND,
                markerSize=18,
                thickness=2,
            )
            cv2.putText(
                annotated,
                "ego-nearest",
                (nearest_point[0] + 7, nearest_point[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )


else:

    print("No segmentation mask")



# ==========================
# 7. Save
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_file = os.path.join(OUTPUT_DIR, "rfdetr_bottom_envelope.jpg")
contacts_file = os.path.join(OUTPUT_DIR, "rfdetr_ground_contacts.json")


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
