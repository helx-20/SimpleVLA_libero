#!/usr/bin/env python3
"""
SimVLA LIBERO Evaluation Client

Observation format:
1. State: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
2. Action: delta action (7D)
3. Default delta control mode
4. Images rotated 180 degrees
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional


import imageio
import json_numpy
import numpy as np
import requests
from tqdm import tqdm

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as ws_client
    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Max steps per task suite (based on longest demo + buffer)
MAX_STEPS = {
    "libero_spatial": 800,   # longest demo: 193
    "libero_object": 800,    # longest demo: 254
    "libero_goal": 800,      # longest demo: 270
    "libero_10": 900,        # longest demo: 505
    "libero_90": 900,        # longest demo: 373
}

NUM_STEPS_WAIT = 10  # Wait for objects to stabilize

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to axis-angle representation.
    
    Uses the same convention as robosuite for consistency with training data.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# -----------------------------------------------------------------------------
# Client Policy Classes
# -----------------------------------------------------------------------------

class WebSocketClient:
    """
    WebSocket client for SimVLA server.
    
    Requires: pip install openpi-client
    """
    def __init__(self, host: str, port: int, replan_steps: int = 5, resize_size: int = 224):
        if not HAS_WS_CLIENT:
            raise ImportError("openpi_client not installed. Run: pip install openpi-client")
        self.client = ws_client.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            # Preprocess images
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["image"], self.resize_size, self.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], self.resize_size, self.resize_size)
            )
            
            # Build observation dict
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
            }
            
            # Query server
            result = self.client.infer(element)
            action_chunk = result["actions"]
            
            # Ensure numpy array
            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)
            
            assert len(action_chunk) >= self.replan_steps, \
                f"Need {self.replan_steps} steps but got {len(action_chunk)}"
            
            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])

        return self.action_plan.popleft()


class LocalClient:
    """In-process SimVLA policy — no WebSocket server required.

    Combines the official paths verbatim:
      * Client-side preprocessing comes from :class:`WebSocketClient` -
        ``image_tools.resize_with_pad`` + ``convert_to_uint8`` to 224 px.
      * Model loading + inference mirrors ``serve_smolvlm_libero.py`` -
        same ``SmolVLMVLA.from_pretrained`` / ``SmolVLMVLAProcessor`` /
        ``generate_actions`` calls, same 384-px Resize + ImageNet
        normalization.

    This way the local path is byte-for-byte equivalent to running
    ``serve_smolvlm_libero.py`` behind a WebSocket - no extra dependency
    on ``adversarial_training`` code.
    """

    IMAGE_SIZE = 384
    ACTION_HORIZON = 10
    STATE_DIM = 8
    ACTION_DIM = 7
    DEFAULT_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"

    def __init__(
        self,
        checkpoint: str,
        norm_stats: Optional[str] = None,
        smolvlm_model: Optional[str] = None,
        replan_steps: int = 5,
        resize_size: int = 224,
    ):
        # Lazy heavy imports so libero_client.py stays importable in envs
        # that only need the WebSocket / HTTP clients.
        import torch
        from PIL import Image
        from torchvision import transforms

        # Add code/ to sys.path so ``from models...`` resolves the same way
        # serve_smolvlm_libero.py does.
        _here = Path(__file__).resolve()
        code_root = _here.parents[2]
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        from models.modeling_smolvlm_vla import SmolVLMVLA
        from models.processing_smolvlm_vla import SmolVLMVLAProcessor

        self._torch = torch
        self._Image = Image
        self.replan_steps = int(replan_steps)
        self.resize_size = int(resize_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mirror serve_smolvlm_libero.load_model
        self.model = SmolVLMVLA.from_pretrained(checkpoint).to(self.device).eval()
        smolvlm_path = smolvlm_model or self.DEFAULT_SMOLVLM
        self.processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_path)
        if norm_stats:
            if not os.path.exists(norm_stats):
                # Loud failure: silently skipping norm_stats lets inference run
                # with default/identity normalization, which produces garbage
                # actions and ~0% task success — common debugging trap.
                raise FileNotFoundError(
                    f"--norm_stats path does not exist: {norm_stats}\n"
                    f"  (Refusing to run with unnormalized actions. Either fix "
                    f"the path or omit --norm_stats explicitly.)"
                )
            self.model.action_space.load_norm_stats(norm_stats)

        # Mirror serve_smolvlm_libero.preprocess_images
        self._transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            # Client-side preprocessing — identical to WebSocketClient.step.
            if not HAS_WS_CLIENT:
                raise RuntimeError(
                    "openpi_client not installed. Run: pip install openpi-client"
                )
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["image"], self.resize_size, self.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], self.resize_size, self.resize_size)
            )

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
            }
            action_chunk = self._infer(element)

            assert len(action_chunk) >= self.replan_steps, \
                f"Need {self.replan_steps} steps but got {len(action_chunk)}"
            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])

        return self.action_plan.popleft()

    def _infer(self, element: Dict) -> np.ndarray:
        """Local inference — mirrors serve_smolvlm_libero.infer."""
        torch = self._torch

        # State padding (same as the server)
        state = np.asarray(element["observation/state"], dtype=np.float32)
        if len(state) < self.STATE_DIM:
            state = np.pad(state, (0, self.STATE_DIM - len(state)))
        state = state[: self.STATE_DIM]

        # Image preprocessing 224 -> 384 + ImageNet normalize
        images, image_mask = self._preprocess_images(
            element["observation/image"], element["observation/wrist_image"],
        )
        images = images.to(self.device)
        image_mask = image_mask.to(self.device)

        # Language encoding
        lang = self.processor.encode_language([element["prompt"]])
        lang = {k: v.to(self.device) for k, v in lang.items()}

        # Proprioception
        proprio = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions = self.model.generate_actions(
                input_ids=lang["input_ids"],
                image_input=images,
                image_mask=image_mask,
                proprio=proprio,
                steps=self.ACTION_HORIZON,
            )
        return actions.cpu().numpy()[0]

    def _preprocess_images(self, image0: np.ndarray, image1: np.ndarray):
        """Mirror serve_smolvlm_libero.preprocess_images byte-for-byte."""
        torch = self._torch
        Image = self._Image
        img0 = self._transform(Image.fromarray(image0.astype(np.uint8)))
        img1 = self._transform(Image.fromarray(image1.astype(np.uint8)))
        padding = torch.zeros_like(img0)
        images = torch.stack([img0, img1, padding], dim=0).unsqueeze(0)
        image_mask = torch.tensor([[True, True, False]])
        return images, image_mask


class HTTPClient:
    """
    HTTP client for SimVLA server.
    """
    def __init__(self, host: str, port: int, replan_steps: int = 5):
        self.url = f"http://{host}:{port}/act"
        self.replan_steps = replan_steps
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def infer(self, element: Dict) -> Dict:
        try:
            payload = {
                "image0": json_numpy.dumps(element["observation/image"]),
                "image1": json_numpy.dumps(element["observation/wrist_image"]),
                "proprio": json_numpy.dumps(element["observation/state"]),
                "language_instruction": element["prompt"],
                "steps": 10,
            }
            
            resp = requests.post(self.url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            actions = np.array(data["action"])
            return {"actions": actions}
            
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            element = {
                "observation/image": obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "observation/state": obs["state"],
                "prompt": goal,
            }
            
            result = self.infer(element)
            action_chunk = result["actions"]
            
            for action in action_chunk[:self.replan_steps]:
                self.action_plan.append(action)

        return self.action_plan.popleft()


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def get_libero_env(task, resolution: int, seed: int):
    """Initialize a LIBERO environment."""
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(
    client,
    task_suite_name: str,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
) -> float:
    """
    Run LIBERO evaluation across all tasks in a suite.
    """
    np.random.seed(seed)
    
    # Initialize task suite
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)
    
    Path(video_out_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Task suite: {task_suite_name}")
    print(f"   Tasks: {num_tasks}, Trials per task: {num_trials}")
    print(f"   Max steps: {max_steps}")
    
    total_episodes, total_successes = 0, 0
    
    for task_id in tqdm(range(num_tasks - 1, -1, -1), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        
        task_successes = 0
        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            # Reset
            env.reset()
            client.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            
            replay_images = []
            t = 0
            done = False
            
            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    # Wait for objects to stabilize
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    # Get images (rotated 180 degrees)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    if save_video:
                        replay_images.append(img)
                    
                    # Build state vector
                    # [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ])
                    
                    # Pack observation
                    obs_dict = {
                        "image": img,
                        "wrist_image": wrist_img,
                        "state": state,
                    }
                    
                    # Get action (7D delta action)
                    action = client.step(obs_dict, task_description)
                    
                    # Execute (send delta action directly)
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    t += 1
                    
                except Exception as e:
                    print(f"Error in rollout: {e}")
                    break

            total_episodes += 1
            
            # Save video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:50]
            video_path = Path(video_out_path) / f"{task_segment}_ep{ep}_{suffix}.mp4"
            if replay_images and save_video:
                imageio.mimwrite(str(video_path), replay_images, fps=10)
            
            # Print episode result
            status_icon = "[OK]" if done else "[FAIL]"
            print(f"  {status_icon} Task {task_id} Ep {ep}: {suffix.upper()} (steps={t})")

        env.close()
        print(f"   Task {task_id}: {task_successes}/{num_trials} ({task_successes/num_trials*100:.1f}%)")
    
    success_rate = total_successes / max(total_episodes, 1)
    print(f"\nTotal success rate: {total_successes}/{total_episodes} ({success_rate*100:.1f}%)")
    
    return success_rate


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _load_criticality_scorer(ckpt_path: str):
    """Load the criticality model and return ``(scorer_fn, device)``.

    ``scorer_fn(init_states: np.ndarray) -> np.ndarray``
        Takes ``(N, D)`` init-state vectors and returns ``(N,)`` P(fail).
    """
    import torch
    # The eval client may be launched from any directory; add code/ to path.
    _here = Path(__file__).resolve()
    code_root = _here.parents[2]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from adversarial_training.utils.criticality_model import (
        CriticalityModel, CriticalityModelConfig,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "input_proj.weight" not in state:
        raise RuntimeError(f"No input_proj.weight in {ckpt_path}")
    max_D = int(state["input_proj.weight"].shape[1])
    model = CriticalityModel(CriticalityModelConfig(
        input_dim=max_D,
        hidden_dim=128,
        expansion=1,
        depth=4,
        dropout=0.0,
    )).to(device).eval()
    model.load_state_dict(state)

    def _score(candidates: np.ndarray) -> np.ndarray:
        x = np.asarray(candidates, dtype=np.float32)
        D = x.shape[1]
        if D < max_D:
            pad = np.zeros((x.shape[0], max_D - D), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        elif D > max_D:
            raise ValueError(f"Init state dim {D} > model input_dim {max_D}")
        t = torch.from_numpy(x).to(device)
        return model.criticality_score(t).detach().cpu().numpy()
    return _score


def eval_libero_routed(
    base_client,
    ft_client,
    crit_scorer,
    task_suite_name: str,
    crit_threshold: float = 0.5,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
) -> float:
    """Like :func:`eval_libero` but routes each episode to ``base_client`` or
    ``ft_client`` based on the criticality of its init state.

    Criticality is pre-computed for the official 50 init states per task before
    the episode loop, so routing adds negligible overhead.
    """
    np.random.seed(seed)
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)
    Path(video_out_path).mkdir(parents=True, exist_ok=True)

    print(f"Task suite: {task_suite_name}  (routed: crit > {crit_threshold} → ft)")
    print(f"   Tasks: {num_tasks}, Trials per task: {num_trials}")
    print(f"   Max steps: {max_steps}")

    # Pre-compute criticality for every official init state.
    print("Pre-computing criticality for official init states ...")
    crit_table: Dict[int, np.ndarray] = {}
    route_counts = {"base": 0, "ft": 0}
    for task_id in range(num_tasks):
        inits = task_suite.get_task_init_states(task_id)
        crit_table[task_id] = crit_scorer(inits)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm(range(num_tasks - 1, -1, -1), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        scores = crit_table[task_id]

        task_successes = 0
        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            init_idx = ep % len(initial_states)
            use_ft = bool(scores[init_idx] > crit_threshold)
            client = ft_client if use_ft else base_client
            route_counts["ft" if use_ft else "base"] += 1

            env.reset()
            client.reset()
            obs = env.set_init_state(initial_states[init_idx])

            replay_images = []
            t = 0
            done = False

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    if save_video:
                        replay_images.append(img)

                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ])

                    action = client.step(
                        {"image": img, "wrist_image": wrist_img, "state": state},
                        task_description,
                    )
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    print(f"Error in rollout: {e}")
                    break

            total_episodes += 1
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:50]
            video_path = Path(video_out_path) / f"{task_segment}_ep{ep}_{suffix}.mp4"
            if replay_images and save_video:
                imageio.mimwrite(str(video_path), replay_images, fps=10)
            route_tag = "[ft]" if use_ft else "[base]"
            status_icon = "[OK]" if done else "[FAIL]"
            print(f"  {status_icon} {route_tag} Task {task_id} Ep {ep}: {suffix.upper()} (steps={t})")

        env.close()
        n_base = int((scores <= crit_threshold).sum())
        n_ft = int((scores > crit_threshold).sum())
        print(f"   Task {task_id}: {task_successes}/{num_trials} ({task_successes/num_trials*100:.1f}%)  "
              f"[{n_base} base inits, {n_ft} ft inits]")

    success_rate = total_successes / max(total_episodes, 1)
    print(f"\nTotal success rate: {total_successes}/{total_episodes} ({success_rate*100:.1f}%)")
    print(f"Route stats: {route_counts['base']} base, {route_counts['ft']} ft episodes")
    return success_rate


def _run_one_episode(env, client, initial_state, task_description,
                     max_steps: int, save_video: bool):
    """Run a single episode and return (done: bool, steps: int)."""
    env.reset()
    client.reset()
    obs = env.set_init_state(initial_state)

    replay_images = []
    t = 0
    done = False

    while t < max_steps + NUM_STEPS_WAIT:
        try:
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            if save_video:
                replay_images.append(img)

            state = np.concatenate([
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ])

            action = client.step(
                {"image": img, "wrist_image": wrist_img, "state": state},
                task_description,
            )
            obs, reward, done, info = env.step(action.tolist())

            if done:
                break
            t += 1
        except Exception as e:
            print(f"Error in rollout: {e}")
            break

    return done, t, replay_images


def eval_libero_routed_ab(
    base_client,
    ft_client,
    crit_scorer,
    task_suite_name: str,
    crit_threshold: float = 0.5,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
    trial_start: int = 0,
    trial_end: int | None = None,
    task_start: int | None = None,
    task_end: int | None = None,
) -> Dict:
    """A/B comparison: on crit > threshold episodes, run BOTH base and ft
    on the **same** init state, so we can directly compare performance.

    Returns a dict with per-model and per-route success counts.
    """
    np.random.seed(seed)
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)
    Path(video_out_path).mkdir(parents=True, exist_ok=True)

    print(f"Task suite: {task_suite_name}  (A/B: crit > {crit_threshold} → both models)")
    print(f"   Tasks: {num_tasks}, Trials per task: {num_trials}")
    print(f"   Max steps: {max_steps}")

    # Pre-compute criticality
    print("Pre-computing criticality for official init states ...")
    crit_table: Dict[int, np.ndarray] = {}
    for task_id in range(num_tasks):
        inits = task_suite.get_task_init_states(task_id)
        crit_table[task_id] = crit_scorer(inits)

    # Counters
    ft_ok = ft_total = 0           # FT on crit>threshold episodes
    base_on_hard_ok = base_on_hard_total = 0  # Base on SAME crit>threshold episodes
    base_on_easy_ok = base_on_easy_total = 0  # Base on crit≤threshold episodes

    t_start = task_start if task_start is not None else 0
    t_end = task_end if task_end is not None else num_tasks
    task_ids = list(range(t_end - 1, t_start - 1, -1))
    for task_id in tqdm(task_ids, desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        scores = crit_table[task_id]

        task_ft_ok = task_ft_total = 0
        task_base_on_hard_ok = task_base_on_hard_total = 0
        task_base_on_easy_ok = task_base_on_easy_total = 0

        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            init_idx = ep % len(initial_states)
            init_state = initial_states[init_idx]
            is_hard = bool(scores[init_idx] > crit_threshold)

            if is_hard:
                # ── Hard episode: run BOTH models on same init state ──
                # FT model
                ft_done, ft_steps, _ = _run_one_episode(
                    env, ft_client, init_state, task_description, max_steps, save_video,
                )
                ft_total += 1
                task_ft_total += 1
                if ft_done:
                    ft_ok += 1
                    task_ft_ok += 1

                # Base model (same init state)
                base_done, base_steps, _ = _run_one_episode(
                    env, base_client, init_state, task_description, max_steps, save_video,
                )
                base_on_hard_total += 1
                task_base_on_hard_total += 1
                if base_done:
                    base_on_hard_ok += 1
                    task_base_on_hard_ok += 1

                ft_icon = "[OK]" if ft_done else "[FAIL]"
                base_icon = "[OK]" if base_done else "[FAIL]"
                print(f"  {ft_icon}[ft] {base_icon}[base] Task {task_id} Ep {ep}: "
                      f"FT={'SUCCESS' if ft_done else 'FAILURE'} "
                      f"Base={'SUCCESS' if base_done else 'FAILURE'} "
                      f"(crit={scores[init_idx]:.4f})")
            else:
                # ── Easy episode: base model only ──
                base_done, base_steps, _ = _run_one_episode(
                    env, base_client, init_state, task_description, max_steps, save_video,
                )
                base_on_easy_total += 1
                task_base_on_easy_total += 1
                if base_done:
                    base_on_easy_ok += 1
                    task_base_on_easy_ok += 1

                base_icon = "[OK]" if base_done else "[FAIL]"
                print(f"  {base_icon}[base] Task {task_id} Ep {ep}: "
                      f"{'SUCCESS' if base_done else 'FAILURE'} "
                      f"(steps={base_steps}, crit={scores[init_idx]:.4f})")

        env.close()
        print(f"   Task {task_id}: "
              f"FT={task_ft_ok}/{task_ft_total} "
              f"Base_on_hard={task_base_on_hard_ok}/{task_base_on_hard_total} "
              f"Base_on_easy={task_base_on_easy_ok}/{task_base_on_easy_total}")

    # ── Summary ──
    n_hard = ft_total
    n_easy = base_on_easy_total
    ft_rate = ft_ok / max(ft_total, 1)
    base_hard_rate = base_on_hard_ok / max(base_on_hard_total, 1)
    base_easy_rate = base_on_easy_ok / max(base_on_easy_total, 1)
    base_overall_rate = (base_on_hard_ok + base_on_easy_ok) / max(base_on_hard_total + base_on_easy_total, 1)
    routed_rate = (ft_ok + base_on_easy_ok) / max(ft_total + base_on_easy_total, 1)

    print(f"\n{'='*60}")
    print(f"A/B Comparison Results — {task_suite_name}")
    print(f"{'='*60}")
    print(f"  Easy episodes (crit≤{crit_threshold}): {n_easy}")
    print(f"    Base only:  {base_on_easy_ok}/{base_on_easy_total} ({base_easy_rate*100:.1f}%)")
    print(f"  Hard episodes (crit>{crit_threshold}): {n_hard}")
    print(f"    FT:         {ft_ok}/{ft_total} ({ft_rate*100:.1f}%)")
    print(f"    Base:       {base_on_hard_ok}/{base_on_hard_total} ({base_hard_rate*100:.1f}%)")
    print(f"  ---")
    print(f"  Base overall: {(base_on_hard_ok+base_on_easy_ok)}/{base_on_hard_total+base_on_easy_total} ({base_overall_rate*100:.1f}%)")
    print(f"  Routed (FT on hard, Base on easy): {ft_ok+base_on_easy_ok}/{ft_total+base_on_easy_total} ({routed_rate*100:.1f}%)")

    return {
        "suite": task_suite_name,
        "n_easy": n_easy, "n_hard": n_hard,
        "ft_ok": ft_ok, "ft_total": ft_total, "ft_rate": ft_rate,
        "base_on_hard_ok": base_on_hard_ok, "base_on_hard_total": base_on_hard_total,
        "base_on_hard_rate": base_hard_rate,
        "base_on_easy_ok": base_on_easy_ok, "base_on_easy_total": base_on_easy_total,
        "base_on_easy_rate": base_easy_rate,
        "base_overall_rate": base_overall_rate,
        "routed_rate": routed_rate,
    }


def main():
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server connection info JSON")
    parser.add_argument("--client_type", type=str, default="websocket",
                        choices=["websocket", "http", "local"],
                        help="Client type: websocket | http | local (in-process model)")
    # local-only:
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[local] SimVLA checkpoint path or HF id")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="[local] Path to norm-stats JSON")
    parser.add_argument("--smolvlm_model", type=str, default=None,
                        help="[local] SmolVLM backbone path or HF id "
                             "(default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--video_out", type=str, default="./eval_results")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording for faster evaluation")
    # ── Routed (dual-model) evaluation ──
    parser.add_argument("--ft_port", type=int, default=None,
                        help="Port of the fine-tuned model server for routed eval. "
                             "When set, --port is the base model. Requires --criticality_ckpt.")
    parser.add_argument("--ft_checkpoint", type=str, default=None,
                        help="[local] Fine-tuned checkpoint for routed eval.")
    parser.add_argument("--criticality_ckpt", type=str, default=None,
                        help="Criticality model checkpoint for per-episode routing.")
    parser.add_argument("--criticality_threshold", type=float, default=0.5,
                        help="Route to ft model when P(fail|init) > threshold (default: 0.5).")
    parser.add_argument("--ab_compare", action="store_true",
                        help="A/B mode: on crit>threshold episodes, run BOTH base "
                             "and ft on the same init state for direct comparison.")
    parser.add_argument("--trial_start", type=int, default=0,
                        help="First trial index (for parallel sharding across GPUs).")
    parser.add_argument("--trial_end", type=int, default=None,
                        help="Last trial index (exclusive). Default: num_trials.")
    parser.add_argument("--task_start", type=int, default=None,
                        help="First task index (for parallel sharding). 0-based, inclusive.")
    parser.add_argument("--task_end", type=int, default=None,
                        help="Last task index (exclusive). Default: n_tasks.")

    args = parser.parse_args()

    # ── CUDA determinism (must be set BEFORE any torch import) ──
    import os as _os
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch as _torch
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
        if hasattr(_torch, "use_deterministic_algorithms"):
            _torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    # Load connection info (only meaningful for websocket/http)
    if args.client_type != "local" and args.connection_info:
        print(f"Loading connection info from: {args.connection_info}")
        while not Path(args.connection_info).exists():
            sys.stdout.write("\rWaiting for server...")
            sys.stdout.flush()
            time.sleep(0.5)
        print()
        with open(args.connection_info) as f:
            info = json.load(f)
            args.host = info["host"]
            args.port = info["port"]

    print(f"Starting LIBERO evaluation client")
    print(f"   Client type: {args.client_type}")
    if args.client_type == "local":
        print(f"   Checkpoint: {args.checkpoint}")
        print(f"   Norm stats: {args.norm_stats}")
    else:
        protocol = "ws" if args.client_type == "websocket" else "http"
        print(f"   Server: {protocol}://{args.host}:{args.port}")
    print(f"   Task suite: {args.task_suite}")
    print(f"   Replan steps: {args.replan_steps}")
    print()

    # Initialize client
    use_routed = (args.ft_port is not None) or (args.ft_checkpoint is not None)
    if use_routed:
        if not args.criticality_ckpt:
            raise SystemExit("--ft_port/--ft_checkpoint requires --criticality_ckpt")
        crit_scorer = _load_criticality_scorer(args.criticality_ckpt)
        print(f"   Criticality ckpt: {args.criticality_ckpt}")
        print(f"   Threshold: {args.criticality_threshold}")

    if args.client_type == "websocket":
        base_client = WebSocketClient(args.host, args.port, replan_steps=args.replan_steps)
        if use_routed:
            ft_client = WebSocketClient(args.host, args.ft_port, replan_steps=args.replan_steps)
            client_kwargs = {
                "base_client": base_client,
                "ft_client": ft_client,
                "crit_scorer": crit_scorer,
                "crit_threshold": args.criticality_threshold,
            }
        else:
            client_kwargs = {"client": base_client}
    elif args.client_type == "http":
        base_client = HTTPClient(args.host, args.port, replan_steps=args.replan_steps)
        if use_routed:
            ft_client = HTTPClient(args.host, args.ft_port, replan_steps=args.replan_steps)
            client_kwargs = {
                "base_client": base_client,
                "ft_client": ft_client,
                "crit_scorer": crit_scorer,
                "crit_threshold": args.criticality_threshold,
            }
        else:
            client_kwargs = {"client": base_client}
    else:  # local
        if not args.checkpoint:
            raise SystemExit("--client_type local requires --checkpoint")
        base_client = LocalClient(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            replan_steps=args.replan_steps,
        )
        if use_routed:
            if not args.ft_checkpoint:
                raise SystemExit("--ft_port/--ft_checkpoint requires --ft_checkpoint for local mode")
            ft_client = LocalClient(
                checkpoint=args.ft_checkpoint,
                norm_stats=args.norm_stats,
                smolvlm_model=args.smolvlm_model,
                replan_steps=args.replan_steps,
            )
            client_kwargs = {
                "base_client": base_client,
                "ft_client": ft_client,
                "crit_scorer": crit_scorer,
                "crit_threshold": args.criticality_threshold,
            }
        else:
            client_kwargs = {"client": base_client}

    # Run evaluation
    video_path = Path(args.video_out) / args.task_suite
    if args.ab_compare:
        # A/B mode: both models on same hard episodes
        result = eval_libero_routed_ab(
            task_suite_name=args.task_suite,
            num_trials=args.num_trials,
            seed=args.seed,
            video_out_path=str(video_path),
            save_video=not args.no_video,
            trial_start=args.trial_start,
            trial_end=args.trial_end,
            task_start=args.task_start,
            task_end=args.task_end,
            **client_kwargs,
        )
    elif use_routed:
        eval_libero_routed(
            task_suite_name=args.task_suite,
            num_trials=args.num_trials,
            seed=args.seed,
            video_out_path=str(video_path),
            save_video=not args.no_video,
            **client_kwargs,
        )
    else:
        eval_libero(
            task_suite_name=args.task_suite,
            num_trials=args.num_trials,
            seed=args.seed,
            video_out_path=str(video_path),
            save_video=not args.no_video,
            **client_kwargs,
        )


if __name__ == "__main__":
    main()
