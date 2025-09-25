import dataclasses

import einops
import numpy as np
import cv2

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BimanualInputs(transforms.DataTransformFn):

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # First, concatenate the joints and gripper into the state vector.
        # Pad to the expected input dimensionality of the model (same as action_dim).
        #state = np.concatenate([data["joints"], data["gripper"]])
        state = transforms.pad_to_dim(data['state'], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        front_head = _parse_image(data["front_head"])
        left_hand = _parse_image(data["left_hand"])
        right_hand = _parse_image(data["right_hand"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front_head,
                "left_wrist_0_rgb": left_hand,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": right_hand,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                #"right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension.
        if "actions" in data:
            # The robot produces 16D actions (14 DoF + 2 gripper), and we pad these.
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 16 action dimensions (6 DoF + gripper), return the first 16 dims
        return {"actions": np.asarray(data["actions"][:, :16])}