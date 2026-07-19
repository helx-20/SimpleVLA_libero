"""Local in-process SimpleVLA policy wrapper.

Replicates the inference path in ``evaluation/libero/serve_smolvlm_libero.py``
but without the WebSocket layer — we load the model once and call it
directly. One ``SimpleVLAPolicy`` instance is created per worker.

Outside the data-collection use case, you can also use this from
accelerated testing (test_model.py) — the interface is the same.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
from openpi_client import image_tools
from PIL import Image
from torchvision import transforms


# Add the SimpleVLA source root to sys.path on import so ``models.*`` resolves
# regardless of where the entry-point script is launched from.
import sys
_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]   # .../
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from models.modeling_smolvlm_vla import SmolVLMVLA          # noqa: E402
from models.processing_smolvlm_vla import SmolVLMVLAProcessor  # noqa: E402


DEFAULT_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMAGE_SIZE = 384
# Intermediate resize done by the official WebSocket client before sending the
# image to the server (libero_client.WebSocketClient resize_size=224). The
# server then resizes 224 -> IMAGE_SIZE. Going through the same two-step
# pipeline locally keeps inference in-distribution with the eval server.
CLIENT_RESIZE_SIZE = 224
ACTION_HORIZON = 10
STATE_DIM = 8
ACTION_DIM = 7


class SimpleVLAPolicy:
    """Thin wrapper exposing ``reset()`` and ``step(obs, prompt)``."""

    def __init__(
        self,
        checkpoint: str,
        norm_stats: Optional[str] = None,
        smolvlm_model: str = DEFAULT_SMOLVLM,
        device: Optional[str] = None,
        replan_steps: int = 5,
        logstd_init: Optional[float] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.replan_steps = int(replan_steps)

        self.model = SmolVLMVLA.from_pretrained(checkpoint).to(self.device).eval()
        self.processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model)

        if norm_stats and Path(norm_stats).exists():
            self.model.action_space.load_norm_stats(norm_stats)

        # BC checkpoints don't train ``actor_logstd``; ``from_pretrained``
        # falls back to the config default (init_logstd = -1.0, σ ≈ 0.37 in
        # normalized action space), which is far too much noise for PPO
        # rollout collection — nearly every episode fails. Override to a
        # tight std here so rollouts stay near the base policy.
        if logstd_init is not None:
            with torch.no_grad():
                self.model.actor_logstd.data.fill_(float(logstd_init))

        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self._plan: Deque[np.ndarray] = collections.deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._plan.clear()

    def step(self, obs: Dict[str, np.ndarray], prompt: str) -> np.ndarray:
        if not self._plan:
            chunk = self._infer(obs, prompt)
            for a in chunk[: self.replan_steps]:
                self._plan.append(a)
        return self._plan.popleft()

    def step_with_record(
        self,
        obs: Dict[str, np.ndarray],
        prompt: str,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Single-step API for PPO rollout.

        Returns ``(action, decision)``. ``decision`` is ``None`` on steps that
        consume the existing plan, and a dict on steps that triggered a fresh
        ``_infer`` call. The dict has the per-chunk PPO bookkeeping:

            mean_normalized      (num_actions, dim_action)  policy mean
            action_normalized    (num_actions, dim_action)  Gaussian sample
            action_chunk         (num_actions, dim_action)  postprocessed for env
            log_prob             scalar, summed over the full chunk
            value                scalar
            logstd               (num_actions, dim_action)  snapshot
            obs_snapshot         dict of arrays at chunk-start (image/wrist/state)
            task_prompt          str
        """
        if not self._plan:
            decision = self._infer_with_ppo(obs, prompt)
            for a in decision["action_chunk"][: self.replan_steps]:
                self._plan.append(np.asarray(a, dtype=np.float32))
            return self._plan.popleft(), decision
        return self._plan.popleft(), None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _infer(self, obs: Dict[str, np.ndarray], prompt: str) -> np.ndarray:
        images, image_mask = self._preprocess_images(obs["image"], obs["wrist_image"])
        images = images.to(self.device)
        image_mask = image_mask.to(self.device)

        lang = self.processor.encode_language([prompt])
        lang = {k: v.to(self.device) for k, v in lang.items()}

        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape[-1] < STATE_DIM:
            state = np.pad(state, (0, STATE_DIM - state.shape[-1]))
        state = state[:STATE_DIM]
        proprio = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions = self.model.generate_actions(
                input_ids=lang["input_ids"],
                image_input=images,
                image_mask=image_mask,
                proprio=proprio,
                steps=ACTION_HORIZON,
            )
        return actions.cpu().numpy()[0]

    def _infer_with_ppo(self, obs: Dict[str, np.ndarray], prompt: str) -> Dict[str, Any]:
        """Same as ``_infer`` but uses ``act_and_value`` to capture (μ, sample, log_prob, value)."""
        images, image_mask = self._preprocess_images(obs["image"], obs["wrist_image"])
        images = images.to(self.device)
        image_mask = image_mask.to(self.device)

        lang = self.processor.encode_language([prompt])
        lang = {k: v.to(self.device) for k, v in lang.items()}

        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape[-1] < STATE_DIM:
            state = np.pad(state, (0, STATE_DIM - state.shape[-1]))
        state = state[:STATE_DIM]
        proprio = torch.from_numpy(state).unsqueeze(0).to(self.device)

        bundle = self.model.act_and_value(
            input_ids=lang["input_ids"],
            image_input=images,
            image_mask=image_mask,
            proprio=proprio,
            steps=ACTION_HORIZON,
            deterministic=False,
        )

        return {
            "mean_normalized":   bundle["mean_normalized"].squeeze(0).cpu().numpy().astype(np.float32),
            "action_normalized": bundle["action_normalized"].squeeze(0).cpu().numpy().astype(np.float32),
            "action_chunk":      bundle["action"].squeeze(0).cpu().numpy().astype(np.float32),
            "log_prob":          float(bundle["log_prob"].item()),
            "value":             float(bundle["value"].item()),
            "logstd":            bundle["logstd"].cpu().numpy().astype(np.float32),
            "obs_snapshot": {
                "image":       np.asarray(obs["image"], dtype=np.uint8).copy(),
                "wrist_image": np.asarray(obs["wrist_image"], dtype=np.uint8).copy(),
                "state":       np.asarray(state, dtype=np.float32).copy(),
            },
            "task_prompt": prompt,
        }

    def _preprocess_images(self, image0: np.ndarray, image1: np.ndarray):
        img0 = self._transform(_client_resize_pil(image0))
        img1 = self._transform(_client_resize_pil(image1))
        pad = torch.zeros_like(img0)
        images = torch.stack([img0, img1, pad], dim=0).unsqueeze(0)
        mask = torch.tensor([[True, True, False]])
        return images, mask


def _client_resize_pil(img: np.ndarray) -> Image.Image:
    """Mirror the official WebSocket client's image preprocessing byte-for-byte.

    ``libero_client.WebSocketClient.step`` runs
    ``image_tools.resize_with_pad(img, 224, 224)`` followed by
    ``convert_to_uint8`` before sending the image to the server, and the
    server then resizes 224 -> 384. Doing both resizes here (rather than
    256 -> 384 in one shot) keeps inputs in-distribution with eval.

    We call the same two openpi_client helpers — PIL.BILINEAR is *close*
    but its half-pixel offset / coefficient computation differs from
    tf.image.resize, which produced ~0.5pp drift on spatial relative to
    the official eval client.
    """
    arr = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.asarray(img), CLIENT_RESIZE_SIZE, CLIENT_RESIZE_SIZE)
    )
    return Image.fromarray(arr)
