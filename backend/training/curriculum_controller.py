"""
Argus Core v2 - RL-Augmented Curriculum Controller
====================================================
Dynamic curriculum learning via reinforcement learning.

Inspired by:
  - TSRL (Tutor-Student RL, CVPR 2026): Lei et al. — PPO agent dynamically
    reweights training samples based on student state.
  - CRDA (Curriculum RL Data Augmentation, AAAI 2026): RL-guided
    augmentation selection with causal inference.

Our approach: lightweight REINFORCE policy gradient controller that learns
to select which degradations to apply and at what severity, based on
training state (epoch progress, validation AUC, loss trends).

Key differences from TSRL:
  - Controls degradation selection (not sample weights) — aligns with CRDA
  - REINFORCE (not PPO) — simpler, lower overhead
  - State is aggregate (epoch-level), not per-sample — O(N) vs O(N*M)

Reference:
  Lei et al., "Tutor-Student Reinforcement Learning: A Dynamic Curriculum
  for Robust Deepfake Detection", CVPR 2026.
"""

import math
import random
from typing import List, Optional, Tuple, Dict

import numpy as np


class RLCurriculumController:
    """
    RL-based controller for dynamic degradation curriculum.

    Learns a policy π(a|s) that maps training state s to degradation
    selection actions a, optimized via REINFORCE policy gradient.

    State space (4-d):
        [epoch_progress, val_auc, val_auc_delta, avg_trend]

    Action space (12-d):
        [degradation_0 .. degradation_10, severity_multiplier]
        - degradation_i: logit bias for degradation type i (logit → softmax)
        - severity_multiplier: scaling factor for degradation severity (0.5-1.5)

    Reward:
        Δ(val_auc) from previous epoch. If no val_loader, uses Δ(1/train_loss).
    """

    DEGRADATION_TYPES = [
        "jpeg", "gaussian_blur", "motion_blur", "gaussian_noise",
        "s_and_p", "color_jitter", "grayscale", "random_erase",
        "downscale", "cutout", "elastic",
    ]

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.95,
        temperature: float = 1.0,
        temp_decay: float = 0.995,
        min_temp: float = 0.1,
        entropy_coef: float = 0.01,
        baseline_ema: float = 0.9,
    ):
        import torch
        self.torch = torch

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_degradations = len(self.DEGRADATION_TYPES)
        self.action_dim = self.num_degradations + 1

        self.gamma = gamma
        self.temperature = temperature
        self.temp_decay = temp_decay
        self.min_temp = min_temp
        self.entropy_coef = entropy_coef
        self.baseline_ema = baseline_ema

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._policy = self._build_policy_net()
        self._optimizer = torch.optim.AdamW(self._policy.parameters(), lr=lr)
        self._baseline = 0.0

        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []
        self._entropies: List[torch.Tensor] = []

        self._prev_val_auc: Optional[float] = None
        self._prev_train_loss: Optional[float] = None
        self._current_severity_mult = 1.0

    def _build_policy_net(self):
        """Build policy network: state → action logits."""
        return self.torch.nn.Sequential(
            self.torch.nn.Linear(self.state_dim, self.hidden_dim),
            self.torch.nn.ReLU(),
            self.torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            self.torch.nn.ReLU(),
            self.torch.nn.Linear(self.hidden_dim, self.action_dim),
        )

    def _encode_state(
        self,
        epoch: int,
        max_epochs: int,
        val_auc: Optional[float],
        train_loss: Optional[float],
    ) -> np.ndarray:
        """Encode training state as 4-d vector."""
        epoch_progress = min(epoch / max(max_epochs, 1), 1.0)

        if val_auc is not None:
            auc_val = val_auc
            if self._prev_val_auc is not None:
                auc_delta = val_auc - self._prev_val_auc
            else:
                auc_delta = 0.0
        else:
            auc_val = 0.5
            auc_delta = 0.0

        if train_loss is not None and self._prev_train_loss is not None:
            loss_delta = (self._prev_train_loss - train_loss) / max(self._prev_train_loss, 1e-8)
            loss_trend = np.clip(loss_delta, -1.0, 1.0)
        else:
            loss_trend = 0.0

        return np.array([epoch_progress, auc_val, auc_delta, loss_trend], dtype=np.float32)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Sample action from policy given state.

        Returns:
            12-d action vector: [degradation_logits... (11), severity_multiplier (1)]
        """
        import torch

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self._device)
            logits = self._policy(state_t).squeeze(0)

            if deterministic:
                degradation_logits = logits[:self.num_degradations]
                degradation_weights = torch.softmax(degradation_logits, dim=0).cpu().numpy()
                severity_mult = float(torch.sigmoid(logits[-1]).cpu().numpy()) * 1.0 + 0.5
            else:
                temp = max(self.temperature, self.min_temp)
                degradation_logits = logits[:self.num_degradations] / temp
                dist = torch.distributions.Categorical(logits=degradation_logits)
                action_idx = dist.sample()
                one_hot = torch.zeros(self.num_degradations, device=self._device)
                one_hot[action_idx] = 1.0
                degradation_weights = one_hot.cpu().numpy()

                severity_raw = logits[-1] / temp
                severity_noise = torch.randn(1, device=self._device) * 0.1
                severity_logit = severity_raw + severity_noise
                severity_mult = float(torch.sigmoid(severity_logit).cpu().numpy()) * 1.0 + 0.5

                log_prob = dist.log_prob(action_idx)
                entropy = dist.entropy()
                severity_log_prob = -0.5 * ((severity_logit - severity_raw) ** 2).sum()
                self._log_probs.append(log_prob + severity_log_prob * 0.1)
                self._entropies.append(entropy)

        self._current_severity_mult = severity_mult
        return np.concatenate([degradation_weights, np.array([severity_mult])])

    def get_degradation_biases(self) -> np.ndarray:
        """
        Get degradation type biases (for sample weighting in pipeline).

        Returns:
            11-d array where higher = more likely to be selected.
        """
        state_np = self._last_state if hasattr(self, '_last_state') else np.zeros(self.state_dim, dtype=np.float32)
        action = self.get_action(state_np, deterministic=True)
        return action[:self.num_degradations]

    def step(
        self,
        epoch: int,
        max_epochs: int,
        val_auc: Optional[float] = None,
        train_loss: Optional[float] = None,
        controller_lr: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Advance controller one epoch: compute reward, update policy.

        Call this after each validation epoch.

        Returns:
            Dict with training metrics (loss, entropy, temperature, baseline)
        """
        state = self._encode_state(epoch, max_epochs, val_auc, train_loss)
        self._last_state = state

        info: Dict[str, float] = {}
        info["temperature"] = self.temperature
        info["baseline"] = self._baseline
        info["severity_mult"] = self._current_severity_mult

        if len(self._log_probs) > 0:
            reward = self._compute_reward(val_auc, train_loss)
            self._rewards.append(reward)
            info["reward"] = reward

            if len(self._rewards) >= 2:
                policy_loss = self._update_policy()
                info["policy_loss"] = policy_loss

        if val_auc is not None:
            self._prev_val_auc = val_auc
        if train_loss is not None:
            self._prev_train_loss = train_loss

        self.temperature = max(self.temperature * self.temp_decay, self.min_temp)

        return info

    def _compute_reward(
        self,
        val_auc: Optional[float],
        train_loss: Optional[float],
    ) -> float:
        """Compute reward from validation AUC improvement."""
        if val_auc is not None and self._prev_val_auc is not None:
            delta = val_auc - self._prev_val_auc
            reward = delta * 10.0
            reward = np.clip(reward, -2.0, 2.0)
            return reward
        elif train_loss is not None and self._prev_train_loss is not None:
            improvement = (self._prev_train_loss - train_loss) / max(self._prev_train_loss, 1e-8)
            reward = float(np.clip(improvement * 2.0, -1.0, 1.0))
            return reward
        return 0.0

    def _update_policy(self) -> float:
        """REINFORCE policy gradient update with baseline."""
        import torch

        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, device=self._device)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(self._log_probs)
        entropies_t = torch.stack(self._entropies)

        advantages = returns_t - self._baseline
        policy_loss = -(log_probs_t * advantages.detach()).mean()
        entropy_bonus = -self.entropy_coef * entropies_t.mean()
        total_loss = policy_loss + entropy_bonus

        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=1.0)
        self._optimizer.step()

        with torch.no_grad():
            self._baseline = self.baseline_ema * self._baseline + (1 - self.baseline_ema) * returns_t.mean().item()

        loss_val = float(total_loss.detach().cpu().numpy())
        self._log_probs.clear()
        self._rewards.clear()
        self._entropies.clear()

        return loss_val

    def get_severity_multiplier(self) -> float:
        """Get current severity multiplier learned by policy."""
        return self._current_severity_mult

    def reset(self):
        """Reset controller state for new training run."""
        self._log_probs.clear()
        self._rewards.clear()
        self._entropies.clear()
        self._prev_val_auc = None
        self._prev_train_loss = None
        self._baseline = 0.0

    def state_dict(self) -> dict:
        return {
            "policy_state": self._policy.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "baseline": self._baseline,
            "temperature": self.temperature,
        }

    def load_state_dict(self, state_dict: dict):
        self._policy.load_state_dict(state_dict["policy_state"])
        self._optimizer.load_state_dict(state_dict["optimizer_state"])
        self._baseline = state_dict.get("baseline", 0.0)
        self.temperature = state_dict.get("temperature", 1.0)
