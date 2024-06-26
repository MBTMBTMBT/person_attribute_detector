from typing import Tuple, List, Dict, Any

import numpy as np
import re
import tensorflow as tf
from numpy import ndarray
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

model = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16))


def snake_to_camel(snake_str):
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_to_snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def detect(img: np.ndarray, request_masks: list, request_poses: list, confidence: float = 0.5) -> tuple[
    list[dict[str, Any]], list[dict[str, list[int] | Any]]]:
    img = img.astype(np.float32)

    result = model.predict_single(img)

    mask = result.get_mask(threshold=confidence)

    # colored mask (separate colour for each body part)
    # colored_mask = result.get_colored_part_mask(mask)
    # from pathlib import Path
    # output_path = Path('./data/example-output')
    # output_path.mkdir(parents=True, exist_ok=True)
    # tf.keras.preprocessing.image.save_img(
    #     f'{output_path}/output-colored-mask.jpg',
    #     colored_mask
    # )

    # construct masks response
    output_masks = []
    for masks_request in request_masks:
        part_mask = result.get_part_mask(
            mask=tf.identity(mask), part_names=masks_request
        ).squeeze()

        bodypix_mask = {'mask': part_mask, 'shape': part_mask.shape}

        output_masks.append(bodypix_mask)

    # construct poses response and neck coordinates
    poses = result.get_poses()
    output_poses = []

    for pose in poses:
        for i, keypoint in pose.keypoints.items():
            if keypoint.part in request_poses:
                keypoint_msg = {}
                keypoint_msg['xy'] = [int(keypoint.position.x), int(keypoint.position.y)]
                keypoint_msg['score'] = keypoint.score
                keypoint_msg['part'] = keypoint.part
                output_poses.append(keypoint_msg)
        break  # just ignore any poses from more than one person

    return output_masks, output_poses
