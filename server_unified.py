"""
Unified sync and RTC reasoning in one server. 
"""

from flask import Flask, jsonify, request, Response

import argparse
import os
import json
import numpy as np
import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
openpi_client_path = os.path.join(root_path,'packages/openpi-client/src/')
openpi_path = os.path.join(root_path,'src/')
sys.path.append(root_path)
sys.path.append(openpi_path)
sys.path.append(openpi_client_path)
import torch
import cv2
import time, math

from openpi.training import config as _config
from openpi.shared import download
from openpi.policies import policy_config as _policy_config

TARGET_IMG_SIZE = 334  # NOTE need to be consistent with that in calvin2json.py

class LLMRobotServer:
    def __init__(self,):
        # config = _config.get_config("pi0_bimanual_tool")
        # config = _config.get_config("pi0_bimanual_right")
        # config = _config.get_config("pi0_0826_tool_all_single_item_tasks")
        config = _config.get_config("pi05_right")

        # checkpoint_dir = download.maybe_download("/liujinxin/code/lhc/openpi/checkpoints/pi0_bimanual_tool/tool_filtered_678/15000")
        # checkpoint_dir = download.maybe_download("/liujinxin/code/lhc/openpi/checkpoints/pi0_bimanual_right/tool_right/30000")
        checkpoint_dir = download.maybe_download("checkpoints/pi05_right/pi05_test_right/20000")

        # Create a trained policy.
        self.policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        self.warmup_inference()

    # for warmup the inference precompilation.
    def warmup_inference(self):
        head_img = cv2.imread("/liujinxin/code/lhc/wy/openpi/rtc_debug/front_head_img.png")
        left_img = cv2.imread("/liujinxin/code/lhc/wy/openpi/rtc_debug/left_hand_img.png")
        right_img = cv2.imread("/liujinxin/code/lhc/wy/openpi/rtc_debug/right_hand_img.png")
        state = np.load("/liujinxin/code/lhc/wy/openpi/rtc_debug/state.npy")

        instruction = "Put the round tape on the hook"
        delay = 6
        action_exec_s = 10

        data = {
            "prompt": f"{instruction}",
            # "state": torch.from_numpy(state).to(dtype=torch.bfloat16),
            "state": torch.from_numpy(state).to(dtype=torch.float32),
            # "state": torch.from_numpy(state).to(dtype=torch.float32).zero_(),
            "front_head": torch.from_numpy(head_img),
            "left_hand": torch.from_numpy(left_img),
            "right_hand": torch.from_numpy(right_img),
        }
        previous_actions = np.zeros((50, 32))
        self.policy.infer_rtc(data, previous_actions=previous_actions, weight_vector=np.zeros(1600))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/zhaowei/workspace/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce12",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--action_stat",
        type=str,
        default="/wangdonglin/calvin/task_ABCD_D/training/statistics.yaml",
    )
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    flask_app = Flask(__name__)
    llm_robot = LLMRobotServer()

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            # added for printing model inference time
            start_time = time.perf_counter_ns()

            front_head = np.frombuffer(request.files["front_head"].read(), dtype=np.uint8)
            front_head = front_head.reshape((224, 224, 3))[:,:,::-1].copy()
            left_hand = np.frombuffer(request.files["left_hand"].read(), dtype=np.uint8)
            left_hand = left_hand.reshape((224, 224, 3))[:,:,::-1].copy()
            right_hand = np.frombuffer(request.files["right_hand"].read(), dtype=np.uint8)
            right_hand = right_hand.reshape((224, 224, 3))[:,:,::-1].copy()

            # cv2.imwrite('img_front_head.jpg', front_head)
            # cv2.imwrite('img_left_hand.jpg', left_hand)
            # cv2.imwrite('img_right_hand.jpg', right_hand)

            state = np.frombuffer(request.files["state"].read(), dtype=np.float64)
            state = state.reshape((16)).copy()

            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]

            data = {
                "prompt": f"{instruction}",
                "state": torch.from_numpy(state).to(dtype=torch.float32),
                "front_head": torch.from_numpy(front_head),
                "left_hand": torch.from_numpy(left_hand),
                "right_hand": torch.from_numpy(right_hand),
            }

            #### added for RTC  #####

            use_rtc = content["use_rtc"]
            if use_rtc:
                previous_actions = np.frombuffer(request.files["previous_actions"].read(), dtype=np.float64)
                # is_first_step = True
                # previous actions need to be padded to (50, 32) shape
                if previous_actions.size != 1:
                    # is_first_step = False
                    previous_actions = previous_actions.reshape((-1, 16))
                    previous_actions = np.pad(previous_actions, ((0, 50 - previous_actions.shape[0]), (0, 16)), 'constant', constant_values=0)

                delay = int(content["delay"])
                action_exec_s = int(content["action_exec_s"])

                ### ----------------- #######
                
                start_time_inference = time.perf_counter_ns()

                # move weight calculation outside of the inference code for compilation requirements
                horizon = 50

                weights = np.zeros(horizon)
                for i in range(horizon):
                    if i < delay:
                        weights[i] = 1
                    elif delay <= i < horizon - action_exec_s:
                        c_i = (horizon - action_exec_s - i) / (horizon - action_exec_s - delay + 1)
                        weights[i] = c_i * (math.exp(c_i) - 1) / (math.exp(1) - 1)
                    else:
                        weights[i] = 0
                weight_vector = np.repeat(weights, 32)

                # print(f"img_static: {img_static.shape}, img_gripper: {img_gripper.shape}, state: {state.shape}")
                result = llm_robot.policy.infer_rtc(data, previous_actions=previous_actions, weight_vector=weight_vector)
                action = result["actions"][None]
                
                # added for printing model inference time
                end_time = time.perf_counter_ns()
                print(f"total inference time: {(end_time - start_time) / 1e6} ms")
                print(f"model inference time: {(end_time - start_time_inference) / 1e6} ms")


                return jsonify(action.tolist())

            else:
                start_time_inference = time.perf_counter_ns()
                result = llm_robot.policy.infer(data)
                action = result["actions"][None]

                # added for printing model inference time
                end_time = time.perf_counter_ns()
                print(f"total inference time: {(end_time - start_time) / 1e6} ms")
                print(f"model inference time: {(end_time - start_time_inference) / 1e6} ms")


                return jsonify(action.tolist())

    flask_app.run(host="0.0.0.0", port=args.port)
