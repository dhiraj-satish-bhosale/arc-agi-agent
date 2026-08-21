"""
Enhanced ARC-AGI-3 Agent — Improved Action Learner
===================================================
An advanced RL-based agent for the ARC-AGI-3 competition.
Builds on the StochasticGoose solution with significant architectural
and algorithmic upgrades.

Key Improvements over the reference solution:
  1. ResNet backbone with Squeeze-and-Excitation (SE) attention blocks
  2. Batch normalization for stable training
  3. Prioritized Experience Replay (PER) for sample efficiency
  4. Focal loss instead of BCE — focuses on hard-to-classify actions
  5. Cosine annealing LR with warm restarts per level
  6. Gradient clipping for training stability
  7. Adaptive entropy regularization (decays over time)
  8. Stagnation detection with exploration boost
  9. AdaptiveAvgPool in action head (eliminates massive FC bottleneck)
  10. Temperature-scaled sampling for controllable exploration

Architecture:
  Input: 16-channel one-hot encoded frames (64x64)
  Backbone: 3-stage ResNet with SE blocks (32->64->128 channels)
  Action Head: AdaptiveAvgPool -> FC(128,64) -> FC(64,5) for ACTION1-5
  Coordinate Head: Conv(128->64->32->1) for 64×64 click positions (ACTION6)
  Total output: 5 + 4096 = 4101 logits

Training:
  Supervised online learning: (state, action) -> frame_changed labels
  Prioritized replay with importance sampling correction
  Focal loss with adaptive entropy regularization
"""

import random
import time
from typing import Any
import numpy as np
import sys
import os
import logging
import hashlib
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -- Framework imports ------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ARC-AGI-3-Agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'arc-prize-2026', 'ARC-AGI-3-Agents'))
from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import setup_experiment_directory, setup_logging_for_experiment, get_environment_directory


# =======================================================================
#  MODEL COMPONENTS
# =======================================================================

class ResBlock(nn.Module):
    """Residual block with batch normalization.
    
    Skip connections allow gradient flow through deeper networks,
    and batch norm stabilizes training — both critical for learning
    from the small, non-stationary data stream during live gameplay.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention.
    
    Learns which color channels (of the 16 ARC colors) are most
    informative for the current game, dynamically re-weighting
    feature maps. This is especially useful because different games
    use different subsets of the 16 available colors.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 4))
        self.fc2 = nn.Linear(max(channels // reduction, 4), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.view(b, c, 1, 1)


class EnhancedActionModel(nn.Module):
    """CNN that predicts which actions will cause frame changes.
    
    Three-stage backbone with residual connections and SE attention,
    followed by two output heads:
      - Action head: predicts ACTION1-5 change probabilities
      - Coordinate head: predicts 64×64 click position probabilities
    
    Uses AdaptiveAvgPool in the action head instead of the reference
    solution's flattened FC layer, reducing parameters from ~34M to ~500K.
    """

    def __init__(self, input_channels: int = 16, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5  # ACTION1-ACTION5

        # -- Stage 1: input_channels -> 32 ----------------------------
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResBlock(32)
        self.se1 = SEBlock(32)

        # -- Stage 2: 32 -> 64 ----------------------------------------
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = ResBlock(64)
        self.se2 = SEBlock(64)

        # -- Stage 3: 64 -> 128 ---------------------------------------
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.res3 = ResBlock(128)
        self.se3 = SEBlock(128)

        # -- Action Head (5 logits) -----------------------------------
        # AdaptiveAvgPool collapses spatial dims -> only 128 features
        # This is far more parameter-efficient than the reference's
        # 65536->512 FC layer.
        self.action_pool = nn.AdaptiveAvgPool2d(1)
        self.action_fc1 = nn.Linear(128, 64)
        self.action_fc2 = nn.Linear(64, self.num_action_types)
        self.dropout = nn.Dropout(0.3)

        # -- Coordinate Head (64×64 logits) ---------------------------
        # Fully convolutional — preserves 2D spatial bias for click
        # position prediction, no flattening until the final output.
        self.coord_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.coord_bn1 = nn.BatchNorm2d(64)
        self.coord_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.coord_bn2 = nn.BatchNorm2d(32)
        self.coord_conv3 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, 64, 64)

        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))     # (batch, 32, 64, 64)
        x = self.se1(self.res1(x))

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))     # (batch, 64, 64, 64)
        x = self.se2(self.res2(x))

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))     # (batch, 128, 64, 64)
        features = self.se3(self.res3(x))

        # Action head
        action_feat = self.action_pool(features).flatten(1)   # (batch, 128)
        action_feat = F.relu(self.action_fc1(action_feat))    # (batch, 64)
        action_feat = self.dropout(action_feat)
        action_logits = self.action_fc2(action_feat)          # (batch, 5)

        # Coordinate head
        coord = F.relu(self.coord_bn1(self.coord_conv1(features)))  # (batch, 64, 64, 64)
        coord = F.relu(self.coord_bn2(self.coord_conv2(coord)))     # (batch, 32, 64, 64)
        coord_logits = self.coord_conv3(coord)                       # (batch, 1, 64, 64)
        coord_logits = coord_logits.view(coord_logits.size(0), -1)   # (batch, 4096)

        # Combined output: [5 action logits, 4096 coordinate logits]
        return torch.cat([action_logits, coord_logits], dim=1)       # (batch, 4101)


# =======================================================================
#  PRIORITIZED EXPERIENCE REPLAY
# =======================================================================

class PrioritizedReplayBuffer:
    """Experience buffer with priority-based sampling.
    
    Instead of uniform random sampling, experiences with higher
    prediction error are sampled more frequently. This dramatically
    improves sample efficiency — the model focuses its limited
    training budget on the most informative experiences.
    
    Uses importance sampling weights to correct for the biased
    sampling distribution, ensuring unbiased gradient updates.
    
    Args:
        maxlen: Maximum buffer capacity
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
    """

    def __init__(self, maxlen: int = 200_000, alpha: float = 0.6):
        self.maxlen = maxlen
        self.alpha = alpha
        self.buffer: list[dict] = []
        self.priorities = np.zeros(maxlen, dtype=np.float64)
        self.position = 0
        self.size = 0
        self.hashes: set[str] = set()
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(self, experience: dict, exp_hash: str) -> bool:
        """Add experience if not a duplicate. Returns True if added."""
        if exp_hash in self.hashes:
            return False

        if self.size < self.maxlen:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority ** self.alpha
        self.hashes.add(exp_hash)
        self.position = (self.position + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)
        return True

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a prioritized batch with importance sampling weights.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full)
        
        Returns:
            (batch, indices, weights) or (None, None, None) if insufficient data
        """
        if self.size < batch_size:
            return None, None, None

        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()

        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)

        # Importance sampling weights correct for biased sampling
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Update priorities based on prediction errors."""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def clear(self):
        """Clear all experiences and reset."""
        self.buffer.clear()
        self.priorities[:] = 0
        self.position = 0
        self.size = 0
        self.hashes.clear()
        self.max_priority = 1.0


# =======================================================================
#  ENHANCED AGENT
# =======================================================================

class Enhanced(Agent):
    """Enhanced RL agent for ARC-AGI-3 with advanced exploration.
    
    Core loop:
      1. Observe frame -> encode as 16-channel one-hot tensor
      2. Feed to CNN -> get action + coordinate logits
      3. Sample action proportional to sigmoid probabilities
      4. Execute action -> observe if frame changed
      5. Store (state, action, reward) in prioritized buffer
      6. Periodically train CNN on prioritized mini-batches
      7. If score increases -> reset model for new level
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # -- Seeding --------------------------------------------------
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.start_time = time.time()

        # No max action limit — run until time expires
        self.MAX_ACTIONS = float('inf')

        # -- Device & Optimization Settings ---------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Enhanced agent using device: {self.device}")

        # Optimize for CPU: if no CUDA, reduce train frequency and batch size to avoid severe FPS drops
        if self.device.type == 'cpu':
            self.batch_size = 32
            self.train_frequency = 20
            print("CPU detected: Optimized batch_size=32, train_frequency=20")
        else:
            self.batch_size = 64
            self.train_frequency = 5
            
        # -- Experiment directory & logging ---------------------------
        self.base_dir, log_file = setup_experiment_directory()
        setup_logging_for_experiment(log_file)
        env_dir = get_environment_directory(self.base_dir, self.game_id)
        self.log_dir = env_dir

        # -- TensorBoard ----------------------------------------------
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # -- Logger ---------------------------------------------------
        self.logger = logging.getLogger(f"EnhancedAgent_{self.game_id}")

        # -- Grid & encoding constants --------------------------------
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size  # 4096
        self.num_colours = 16
        self.current_levels_completed = -1
        self.global_train_step = 0

        # -- Model ----------------------------------------------------
        self.action_model = None
        self.optimizer = None
        self.scheduler = None

        # -- Prioritized replay buffer --------------------------------
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=200_000, alpha=0.6)
        self.per_beta_start = 0.4       # IS weight annealing start
        self.per_beta_end = 1.0         # IS weight annealing end
        self.per_beta_steps = 100_000   # Steps to anneal over

        # -- Focal loss parameters ------------------------------------
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

        # -- Entropy regularization (adaptive) ------------------------
        self.base_action_entropy = 0.001
        self.base_coord_entropy = 0.0001

        # -- Stagnation detection -------------------------------------
        self.stagnation_counter = 0
        self.stagnation_threshold = 200  # Steps without new frame
        self.temperature = 1.0           # Sampling temperature
        self.base_temperature = 1.0
        self.max_temperature = 5.0

        # -- State tracking -------------------------------------------
        self.prev_frame = None
        self.prev_action_idx = None
        self.total_new_frames = 0
        self.level_action_count = 0

        # -- Action mapping -------------------------------------------
        self.action_list = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5
        ]

        self.logger.info(f"Enhanced agent initialized for game_id: {self.game_id}")

    # -----------------------------------------------------------------
    #  FRAME ENCODING
    # -----------------------------------------------------------------

    def _frame_to_tensor(self, frame_data: FrameData) -> torch.Tensor:
        """Convert frame to 16-channel one-hot tensor (16, 64, 64)."""
        frame = np.array(frame_data.frame, dtype=np.int64)
        frame = frame[-1]  # Last frame if animation sequence
        assert frame.shape == (self.grid_size, self.grid_size), \
            f"Expected ({self.grid_size}, {self.grid_size}), got {frame.shape}"

        tensor = torch.zeros(
            self.num_colours, self.grid_size, self.grid_size,
            dtype=torch.float32
        )
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        return tensor.to(self.device)

    # -----------------------------------------------------------------
    #  EXPERIENCE HASHING (deduplication)
    # -----------------------------------------------------------------

    def _compute_experience_hash(self, frame: np.ndarray, action_idx: int) -> str:
        """Hash frame+action for deduplication."""
        hash_input = frame.tobytes() + str(action_idx).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()

    # -----------------------------------------------------------------
    #  FOCAL LOSS
    # -----------------------------------------------------------------

    def _focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss — down-weights easy examples, focuses on hard ones.
        
        Standard BCE treats all examples equally. In ARC-AGI-3, most
        actions DON'T change the frame, so negative examples dominate.
        Focal loss reduces the loss contribution from well-classified
        negatives and focuses the model on the rare positives.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce)  # Probability of correct classification
        focal = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce

        # Apply importance sampling weights from PER
        weighted_focal = focal * weights.to(self.device)
        return weighted_focal.mean()

    # -----------------------------------------------------------------
    #  TRAINING
    # -----------------------------------------------------------------

    def _train_action_model(self):
        """Train on a prioritized mini-batch with focal loss."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Anneal PER beta (importance sampling correction)
        beta = min(
            self.per_beta_end,
            self.per_beta_start + (self.per_beta_end - self.per_beta_start)
            * self.level_action_count / max(self.per_beta_steps, 1)
        )

        # Sample prioritized batch
        batch, indices, is_weights = self.replay_buffer.sample(
            self.batch_size, beta=beta
        )
        if batch is None:
            return

        # Prepare tensors
        states = torch.stack([
            torch.from_numpy(exp['state']).float().to(self.device)
            for exp in batch
        ])
        action_indices = torch.tensor(
            [exp['action_idx'] for exp in batch],
            dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [exp['reward'] for exp in batch],
            dtype=torch.float32, device=self.device
        )

        self.optimizer.zero_grad()

        # Forward pass
        combined_logits = self.action_model(states)  # (batch, 4101)

        # Gather logits for selected actions
        selected_logits = combined_logits.gather(
            1, action_indices.unsqueeze(1)
        ).squeeze(1)

        # Focal loss with importance sampling weights
        main_loss = self._focal_loss(selected_logits, rewards, is_weights)

        # -- Adaptive entropy regularization --------------------------
        # Starts high for exploration, decays as the model improves
        decay = max(0.01, 1.0 - self.level_action_count / 50_000)
        action_coeff = self.base_action_entropy * decay
        coord_coeff = self.base_coord_entropy * decay

        all_probs = torch.sigmoid(combined_logits)
        action_entropy = all_probs[:, :5].mean()
        coord_entropy = all_probs[:, 5:].mean()

        total_loss = (
            main_loss
            - action_coeff * action_entropy
            - coord_coeff * coord_entropy
        )

        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # -- Update priorities based on prediction error --------------
        with torch.no_grad():
            predictions = torch.sigmoid(selected_logits)
            errors = (predictions - rewards).abs().cpu().numpy()
            self.replay_buffer.update_priorities(indices, errors)

        # -- TensorBoard Logging --------------------------------------
        self.global_train_step += 1
        if self.writer is not None:
            self.writer.add_scalar('Train/Total_Loss', total_loss.item(), self.global_train_step)
            self.writer.add_scalar('Train/Focal_Loss', main_loss.item(), self.global_train_step)
            self.writer.add_scalar('Train/Action_Entropy', action_entropy.item(), self.global_train_step)
            self.writer.add_scalar('Train/Coord_Entropy', coord_entropy.item(), self.global_train_step)
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_train_step)
            self.writer.add_scalar('Train/PER_Beta', beta, self.global_train_step)

        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    #  ACTION SAMPLING
    # -----------------------------------------------------------------

    def _sample_from_combined_output(
        self, combined_logits: torch.Tensor,
        available_actions: list = None
    ) -> tuple[int, tuple | None, int | None, np.ndarray]:
        """Sample from combined action+coordinate space with masking.
        
        Hierarchical: first decide action type via sigmoid probs,
        then if click (ACTION6) is selected, pick coordinates.
        Temperature scaling controls exploration vs exploitation.
        """
        action_logits = combined_logits[:5]
        coord_logits = combined_logits[5:]

        # -- Mask unavailable actions ---------------------------------
        if available_actions is not None and len(available_actions) > 0:
            action_mask = torch.full_like(action_logits, float('-inf'))
            action6_available = False

            for action in available_actions:
                action_id = action.value if hasattr(action, "value") else int(action)
                if 1 <= action_id <= 5:
                    action_mask[action_id - 1] = 0.0
                elif action_id == 6:
                    action6_available = True

            action_logits = action_logits + action_mask

            if not action6_available:
                coord_logits = coord_logits + torch.full_like(
                    coord_logits, float('-inf')
                )

        # -- Temperature-scaled sigmoid probabilities -----------------
        action_probs = torch.sigmoid(action_logits / self.temperature)
        coord_probs_raw = torch.sigmoid(coord_logits / self.temperature)

        # Scale coordinate probs for fair sampling against actions
        coord_probs_scaled = coord_probs_raw / self.num_coordinates

        # Combine and normalize
        all_probs = torch.cat([action_probs, coord_probs_scaled])
        prob_sum = all_probs.sum()
        if prob_sum > 0:
            all_probs = all_probs / prob_sum
        else:
            # Fallback to uniform if all masked
            all_probs = torch.ones_like(all_probs) / len(all_probs)

        all_probs_np = all_probs.cpu().numpy()

        # Fix any numerical issues
        all_probs_np = np.clip(all_probs_np, 0, None)
        prob_sum = all_probs_np.sum()
        if prob_sum > 0:
            all_probs_np /= prob_sum
        else:
            all_probs_np = np.ones_like(all_probs_np) / len(all_probs_np)

        # Sample
        selected_idx = np.random.choice(len(all_probs_np), p=all_probs_np)

        # Visualization probs (unscaled sigmoid)
        viz_probs = torch.cat([
            torch.sigmoid(action_logits),
            torch.sigmoid(coord_logits)
        ]).cpu().numpy()

        if selected_idx < 5:
            return selected_idx, None, None, viz_probs
        else:
            coord_idx = selected_idx - 5
            y_idx = coord_idx // self.grid_size
            x_idx = coord_idx % self.grid_size
            return 5, (y_idx, x_idx), coord_idx, viz_probs

    # -----------------------------------------------------------------
    #  LEVEL RESET
    # -----------------------------------------------------------------

    def _reset_for_new_level(self):
        """Reset model, optimizer, and buffer for a new level."""
        # Fresh model
        self.action_model = EnhancedActionModel(
            input_channels=self.num_colours,
            grid_size=self.grid_size
        ).to(self.device)

        # Optimizer with cosine annealing LR
        self.optimizer = optim.Adam(
            self.action_model.parameters(), lr=0.0003, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=2000, T_mult=2
        )

        # Clear replay buffer
        self.replay_buffer.clear()

        # Reset tracking
        self.prev_frame = None
        self.prev_action_idx = None
        self.stagnation_counter = 0
        self.temperature = self.base_temperature
        self.level_action_count = 0
        self.total_new_frames = 0

        self.logger.info("Reset model, optimizer, and buffer for new level")
        print("Reset model, optimizer, and buffer for new level")

    # -----------------------------------------------------------------
    #  TIME CHECK
    # -----------------------------------------------------------------

    def _has_time_elapsed(self) -> bool:
        """Check if approaching the 8-hour limit (with 5-min safety buffer)."""
        return (time.time() - self.start_time) >= 8 * 3600 - 5 * 60

    # -----------------------------------------------------------------
    #  AGENT INTERFACE
    # -----------------------------------------------------------------

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Done when we win or time expires."""
        return any([
            latest_frame.state is GameState.WIN,
            self._has_time_elapsed(),
        ])

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action using the enhanced learned policy."""

        # -- Handle level/score changes -------------------------------
        if latest_frame.levels_completed != self.current_levels_completed:
            self.logger.info(
                f"Levels completed: {self.current_levels_completed} -> "
                f"{latest_frame.levels_completed} at action {self.action_counter}"
            )
            print(
                f"Levels completed: {self.current_levels_completed} -> "
                f"{latest_frame.levels_completed} at action {self.action_counter}"
            )
            self._reset_for_new_level()
            self.current_levels_completed = latest_frame.levels_completed

        # -- Handle game-over / not-played states --------------------
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.prev_frame = None
            self.prev_action_idx = None
            action = GameAction.RESET
            action.reasoning = "Game needs reset."
            return action

        # -- Encode current frame -------------------------------------
        current_frame = self._frame_to_tensor(latest_frame)
        if current_frame is None:
            self.prev_frame = None
            self.prev_action_idx = None
            action = random.choice(self.action_list[:5])
            action.reasoning = "Frame encoding failed, random action"
            return action

        # -- Create experience from previous step --------------------
        if self.prev_frame is not None:
            exp_hash = self._compute_experience_hash(
                self.prev_frame, self.prev_action_idx
            )
            current_frame_np = current_frame.cpu().numpy().astype(bool)
            frame_changed = not np.array_equal(self.prev_frame, current_frame_np)

            experience = {
                'state': self.prev_frame,
                'action_idx': self.prev_action_idx,
                'reward': 1.0 if frame_changed else 0.0,
            }
            self.replay_buffer.add(experience, exp_hash)

            # -- Stagnation detection ---------------------------------
            if frame_changed:
                self.stagnation_counter = 0
                self.total_new_frames += 1
                # Cool down temperature when making progress
                self.temperature = max(
                    self.base_temperature,
                    self.temperature * 0.95
                )
            else:
                self.stagnation_counter += 1

            # Boost exploration when stuck
            if self.stagnation_counter > self.stagnation_threshold:
                self.temperature = min(
                    self.temperature * 1.05,
                    self.max_temperature
                )

        # -- Model inference ------------------------------------------
        self.action_model.eval()
        with torch.no_grad():
            combined_logits = self.action_model(current_frame.unsqueeze(0))
            combined_logits = combined_logits.squeeze(0)

            action_idx, coords, coord_idx, all_probs = \
                self._sample_from_combined_output(
                    combined_logits, latest_frame.available_actions
                )

            if action_idx < 5:
                selected_action = self.action_list[action_idx]
                selected_action.reasoning = (
                    f"{selected_action.name} "
                    f"(prob: {all_probs[action_idx]:.3f}, "
                    f"temp: {self.temperature:.2f})"
                )
            else:
                selected_action = GameAction.ACTION6
                y, x = coords
                selected_action.set_data({"x": int(x), "y": int(y)})
                selected_action.reasoning = (
                    f"ACTION6 at ({x}, {y}) "
                    f"(prob: {all_probs[5 + coord_idx]:.3f}, "
                    f"temp: {self.temperature:.2f})"
                )

        # -- Store state for next experience --------------------------
        self.prev_frame = current_frame.cpu().numpy().astype(bool)
        if action_idx < 5:
            self.prev_action_idx = action_idx
        else:
            self.prev_action_idx = 5 + coord_idx

        # -- Train periodically ---------------------------------------
        self.level_action_count += 1
        if self.level_action_count % self.train_frequency == 0:
            self.action_model.train()
            self._train_action_model()

        return selected_action
