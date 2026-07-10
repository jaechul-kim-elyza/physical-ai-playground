import os
import numpy as np
from PIL import Image

import cv2

from nuscenes.nuscenes import NuScenes

from rfdetr import RFDETRSegMedium

import supervision as sv


# ==========================
# 1. nuScenes
# ==========================

DATA_ROOT = "/home/kim/datasets/nuScenes"


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATA_ROOT,
    verbose=True
)


# ==========================
# 2. Load CAM_FRONT image
# ==========================

scene = nusc.scene[0]

sample = nusc.get(
    "sample",
    scene["first_sample_token"]
)


cam_token = sample["data"]["CAM_FRONT"]


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


if detections.mask is not None:

    masks = detections.mask


    for obj_id, mask in enumerate(masks):

        h, w = mask.shape


        bottom_points = []


        # x 방향으로 scan
        for x in range(w):

            ys = np.where(mask[:, x])[0]


            if len(ys) > 0:

                y_bottom = ys.max()

                bottom_points.append(
                    [x, y_bottom]
                )


        bottom_points = np.array(
            bottom_points
        )


        print(
            f"Object {obj_id}:",
            bottom_points.shape
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


else:

    print("No segmentation mask")



# ==========================
# 7. Save
# ==========================

out_file = "rfdetr_bottom_envelope.jpg"


Image.fromarray(
    annotated
).save(out_file)


print("====================")
print("Saved:")
print(out_file)
print("====================")