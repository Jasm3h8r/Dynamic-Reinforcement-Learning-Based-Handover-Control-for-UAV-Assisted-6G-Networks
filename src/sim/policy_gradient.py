# ============================================================
# POLICY GRADIENT METHODS: REINFORCE, A2C, SAC
# Fixed version — numerical stability patches applied
# ============================================================

import numpy as np
from typing import List, Tuple, Optional


class PolicyNetworkBase:
    """Base class for all policy gradient methods."""

    def __init__(self, state_dim: int = 13, action_dim: int = 3,
                 lr: float = 1e-3,          # FIX 1: reduced from 3e-3 → 1e-3
                 gamma: float = 0.97,
                 max_speed: float = 15.0,
                 z_min: float = 30.0, z_max: float = 150.0):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.lr         = lr
        self.gamma      = gamma
        self.max_speed  = max_speed
        self.z_min      = z_min
        self.z_max      = z_max

        self.states:  List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float]      = []
        self.pg_update_interval: int   = 10

        self.episode_returns:     List[float] = []
        self.policy_loss_history: List[float] = []
        self.method_name: str = "Base"

        # FIX 2: tightened grad norm from 50 → 1.0
        self.max_grad_norm: float = 1.0
        self.max_weight:    float = 1e3

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    # ── FIX 3: safe_clip — replaces nan/inf throughout forward passes ──────
    @staticmethod
    def _safe(x: np.ndarray, lo: float = -1e4, hi: float = 1e4) -> np.ndarray:
        return np.clip(np.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo), lo, hi)

    # ── FIX 4: global-norm gradient clipping ──────────────────────────────
    @staticmethod
    def _clip_g(g: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        norm = np.sqrt(np.sum(g ** 2))
        if norm > max_norm:
            g = g * (max_norm / (norm + 1e-8))
        return g

    def select_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def store(self, state: np.ndarray, action: np.ndarray, reward: float):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(np.clip(reward, -10.0, 10.0)))  # FIX 5: clip rewards at intake

    def update(self) -> float:
        raise NotImplementedError

    def _clip_weights(self):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, np.ndarray):
                try:
                    # FIX 6: also replace nan/inf in weights, not just clip
                    v[:] = np.nan_to_num(v, nan=0.0, posinf=self.max_weight,
                                         neginf=-self.max_weight)
                    np.clip(v, -self.max_weight, self.max_weight, out=v)
                except Exception:
                    pass


# ============================================================
# REINFORCE WITH BASELINE
# ============================================================

class PolicyNetworkREINFORCE(PolicyNetworkBase):
    """REINFORCE with baseline."""

    def __init__(self, state_dim: int = 13, action_dim: int = 3, **kwargs):
        # FIX 7: enforce safe lr regardless of what caller passes
        kwargs['lr'] = min(kwargs.get('lr', 1e-3), 1e-3)
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)
        self.method_name   = "REINFORCE"
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)   # FIX 2

        hidden = 32
        self.W1    = np.random.randn(state_dim, hidden) * 0.1
        self.b1    = np.zeros(hidden)
        self.W2_mu = np.random.randn(hidden, action_dim) * 0.05
        self.b2_mu = np.zeros(action_dim)
        self.log_std = np.full(action_dim, np.log(2.0))  # FIX 8: init std=2, not 5

        self.Wv1 = np.random.randn(state_dim, hidden) * 0.1
        self.bv1 = np.zeros(hidden)
        self.Wv2 = np.random.randn(hidden, 1) * 0.05
        self.bv2 = np.zeros(1)

    def _forward_policy(self, s: np.ndarray):
        s  = self._safe(s)
        h  = self._relu(self._safe(s @ self.W1 + self.b1))
        mu = np.tanh(self._safe(h @ self.W2_mu + self.b2_mu)) * self.max_speed
        std = np.exp(np.clip(self.log_std, -3, 2))
        return mu, std, h

    def _forward_value(self, s: np.ndarray) -> Tuple[float, np.ndarray]:
        s  = self._safe(s)
        hv = self._relu(self._safe(s @ self.Wv1 + self.bv1))
        v  = float(self._safe(hv @ self.Wv2 + self.bv2)[0])
        return v, hv

    def select_action(self, state: np.ndarray) -> np.ndarray:
        mu, std, _ = self._forward_policy(state)
        action = mu + std * np.random.randn(self.action_dim)
        return np.clip(action, -self.max_speed, self.max_speed)

    def update(self) -> float:
        if len(self.rewards) < max(2, self.pg_update_interval):
            return 0.0

        # Discounted returns
        G = 0.0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float64)

        # FIX 9: normalise AND hard-clip returns to [-5, 5]
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = np.clip(returns, -5.0, 5.0)

        total_loss = 0.0

        for s, a, G_t in zip(self.states, self.actions, returns):
            mu, std, h = self._forward_policy(s)
            baseline, hv = self._forward_value(s)

            # FIX 10: clip advantage before use
            advantage = float(np.clip(G_t - baseline, -5.0, 5.0))

            log_prob = -0.5 * np.sum(((a - mu) / (std + 1e-8)) ** 2) \
                       - np.sum(np.log(std + 1e-8))
            total_loss += -log_prob * advantage

            # ── Policy head gradients ─────────────────────────────────────
            d_mu = advantage * (a - mu) / (std ** 2 + 1e-8)
            d_mu *= (1 - np.tanh(h @ self.W2_mu + self.b2_mu) ** 2) * self.max_speed
            d_mu  = self._safe(d_mu)

            self.W2_mu += self.lr * self._clip_g(np.outer(h, d_mu))
            self.b2_mu += self.lr * self._clip_g(d_mu)

            d_h      = self._safe(d_mu @ self.W2_mu.T)
            d_h_relu = d_h * (h > 0)
            self.W1  += self.lr * self._clip_g(np.outer(s, d_h_relu))
            self.b1  += self.lr * self._clip_g(d_h_relu)

            # log_std gradient
            d_log_std = advantage * (((a - mu) ** 2 / (std ** 2 + 1e-8)) - 1.0)
            self.log_std += self.lr * self._clip_g(d_log_std)
            self.log_std  = np.clip(self.log_std, -3.0, 1.5)   # FIX 11: tighter range

            # ── Value head gradients ──────────────────────────────────────
            # FIX 12: clamp TD error before squaring
            v_err = float(np.clip(G_t - baseline, -5.0, 5.0))
            self.Wv2 += self.lr * self._clip_g(2 * v_err * hv.reshape(-1, 1))
            self.bv2 += self.lr * self._clip_g(2 * v_err * np.ones(1))
            d_hv = self._safe((2 * v_err * self.Wv2).flatten() * (hv > 0))
            self.Wv1 += self.lr * self._clip_g(np.outer(s, d_hv))
            self.bv1 += self.lr * self._clip_g(d_hv)

        self.episode_returns.append(float(np.sum(self.rewards)))
        self.policy_loss_history.append(total_loss / max(len(self.rewards), 1))
        self._clip_weights()
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        return total_loss / max(len(self.states) or 1, 1)


# ============================================================
# ADVANTAGE ACTOR-CRITIC (A2C)
# ============================================================

class PolicyNetworkA2C(PolicyNetworkBase):
    """Advantage Actor-Critic (A2C) with shared layers."""

    def __init__(self, state_dim: int = 13, action_dim: int = 3, **kwargs):
        kwargs['lr'] = min(kwargs.get('lr', 1e-3), 1e-3)
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)
        self.method_name   = "A2C"
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)

        hidden = 32
        self.W1       = np.random.randn(state_dim, hidden) * 0.1
        self.b1       = np.zeros(hidden)
        self.W_actor  = np.random.randn(hidden, action_dim) * 0.05
        self.b_actor  = np.zeros(action_dim)
        self.log_std  = np.full(action_dim, np.log(2.0))   # FIX 8
        self.W_critic = np.random.randn(hidden, 1) * 0.05
        self.b_critic = np.zeros(1)

        self.entropy_coeff  = 0.01
        self.value_loss_coeff = 0.5
        self.pg_update_interval = kwargs.get('pg_update_interval', 10)

    def _forward_base(self, s: np.ndarray) -> np.ndarray:
        return self._relu(self._safe(s @ self.W1 + self.b1))

    def _forward_actor(self, h: np.ndarray):
        mu  = np.tanh(self._safe(h @ self.W_actor + self.b_actor)) * self.max_speed
        std = np.exp(np.clip(self.log_std, -3, 2))
        return mu, std

    def _forward_critic(self, h: np.ndarray) -> float:
        return float(self._safe(h @ self.W_critic + self.b_critic)[0])

    def select_action(self, state: np.ndarray) -> np.ndarray:
        h  = self._forward_base(state)
        mu, std = self._forward_actor(h)
        return np.clip(mu + std * np.random.randn(self.action_dim),
                       -self.max_speed, self.max_speed)

    def update(self) -> float:
        if len(self.rewards) < max(2, self.pg_update_interval):
            return 0.0

        values = []
        for s in self.states:
            h = self._forward_base(s)
            values.append(self._forward_critic(h))

        # N-step returns
        G = 0.0
        targets = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            targets.insert(0, G)
        targets = np.array(targets, dtype=np.float64)

        # FIX 9: normalise targets and clip
        if targets.std() > 1e-6:
            targets = (targets - targets.mean()) / (targets.std() + 1e-8)
        targets = np.clip(targets, -5.0, 5.0)

        advantages = targets - np.array(values, dtype=np.float64)

        # FIX 13: normalise AND clip advantages separately
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = np.clip(advantages, -3.0, 3.0)

        total_loss = 0.0

        for s, a, adv, target in zip(self.states, self.actions, advantages, targets):
            h        = self._forward_base(s)
            mu, std  = self._forward_actor(h)
            v        = self._forward_critic(h)

            log_prob = -0.5 * np.sum(((a - mu) / (std + 1e-8)) ** 2) \
                       - np.sum(np.log(std + 1e-8))
            entropy  = np.sum(np.log(std + 1e-8)) + \
                       0.5 * self.action_dim * (1 + np.log(2 * np.pi))

            actor_loss  = -log_prob * adv - self.entropy_coeff * entropy
            # FIX 12: clamp TD error before squaring
            td_err      = float(np.clip(target - v, -5.0, 5.0))
            critic_loss = self.value_loss_coeff * td_err ** 2
            total_loss += actor_loss + critic_loss

            # ── Actor gradient ────────────────────────────────────────────
            d_actor  = -adv * (a - mu) / (std ** 2 + 1e-8)
            d_actor *= (1 - np.tanh(h @ self.W_actor + self.b_actor) ** 2) * self.max_speed
            d_actor  = self._safe(d_actor)

            d_log_std = adv * (((a - mu) ** 2 / (std ** 2 + 1e-8)) - 1.0) \
                        - self.entropy_coeff / (std + 1e-8)

            self.W_actor += self.lr * self._clip_g(np.outer(h, d_actor))
            self.b_actor += self.lr * self._clip_g(d_actor)
            self.log_std += self.lr * self._clip_g(d_log_std)
            self.log_std  = np.clip(self.log_std, -3.0, 1.5)

            # ── Critic gradient ───────────────────────────────────────────
            d_critic = -2.0 * td_err
            self.W_critic += self.lr * self._clip_g(d_critic * h.reshape(-1, 1))
            self.b_critic += self.lr * self._clip_g(np.array([d_critic]))

            # ── Shared base gradient ──────────────────────────────────────
            d_h = self._safe(d_actor @ self.W_actor.T +
                             d_critic * self.W_critic.T.flatten())
            d_h_relu = d_h * (h > 0)
            self.W1 += self.lr * self._clip_g(np.outer(s, d_h_relu))
            self.b1 += self.lr * self._clip_g(d_h_relu)

        self.episode_returns.append(float(np.sum(self.rewards)))
        self.policy_loss_history.append(total_loss / max(len(self.rewards), 1))
        self._clip_weights()
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        return total_loss / max(len(self.states) or 1, 1)


# ============================================================
# SOFT ACTOR-CRITIC (SAC)
# ============================================================

class PolicyNetworkSoftA2C(PolicyNetworkBase):
    """Soft Actor-Critic with twin critics, replay buffer, entropy tuning."""

    def __init__(self, state_dim: int = 13, action_dim: int = 3, **kwargs):
        kwargs['lr'] = min(kwargs.get('lr', 1e-3), 1e-3)
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)
        self.method_name   = "SAC"
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)

        hidden = 32
        self.W1       = np.random.randn(state_dim, hidden) * 0.1
        self.b1       = np.zeros(hidden)
        self.W_actor  = np.random.randn(hidden, action_dim) * 0.05
        self.b_actor  = np.zeros(action_dim)
        self.log_std  = np.full(action_dim, np.log(2.0))   # FIX 8
        self.W_critic  = np.random.randn(hidden, 1) * 0.05
        self.b_critic  = np.zeros(1)
        self.W_critic2 = np.random.randn(hidden, 1) * 0.05
        self.b_critic2 = np.zeros(1)

        # Target networks
        self.W_critic_target  = self.W_critic.copy()
        self.b_critic_target  = self.b_critic.copy()
        self.W_critic2_target = self.W_critic2.copy()
        self.b_critic2_target = self.b_critic2.copy()
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()

        self.alpha          = 0.2
        self.target_entropy = -float(action_dim)
        self.log_alpha      = np.log(self.alpha)
        self.alpha_lr       = 1e-4
        self.tau            = 0.01

        self.replay_buffer: List[Tuple] = []
        self.replay_size  = kwargs.get('replay_size', 2000)
        self.batch_size   = kwargs.get('batch_size', 64)
        self.pg_update_interval = kwargs.get('pg_update_interval', 10)

    def _forward_base(self, s: np.ndarray, target: bool = False) -> np.ndarray:
        W1 = self.W1_target if target else self.W1
        b1 = self.b1_target if target else self.b1
        return self._relu(self._safe(s @ W1 + b1))

    def _forward_actor(self, h: np.ndarray):
        mu  = np.tanh(self._safe(h @ self.W_actor + self.b_actor)) * self.max_speed
        std = np.exp(np.clip(self.log_std, -3, 2))
        return mu, std

    def _forward_critic(self, h: np.ndarray, target: bool = False) -> float:
        W = self.W_critic_target if target else self.W_critic
        b = self.b_critic_target if target else self.b_critic
        return float(self._safe(h @ W + b)[0])

    def _forward_critic2(self, h: np.ndarray, target: bool = False) -> float:
        W = self.W_critic2_target if target else self.W_critic2
        b = self.b_critic2_target if target else self.b_critic2
        return float(self._safe(h @ W + b)[0])

    def select_action(self, state: np.ndarray) -> np.ndarray:
        h = self._forward_base(state)
        mu, std = self._forward_actor(h)
        return np.clip(mu + std * np.random.randn(self.action_dim),
                       -self.max_speed, self.max_speed)

    def _update_one(self, s, a, r, td_target):
        """Single-transition update shared by on-policy and off-policy paths."""
        h  = self._forward_base(s)
        mu, std = self._forward_actor(h)
        v1 = self._forward_critic(h)
        v2 = self._forward_critic2(h)

        log_prob = -0.5 * np.sum(((a - mu) / (std + 1e-8)) ** 2) \
                   - np.sum(np.log(std + 1e-8))
        entropy  = -log_prob

        # FIX 12: clamp TD errors before squaring
        err1 = float(np.clip(td_target - v1, -5.0, 5.0))
        err2 = float(np.clip(td_target - v2, -5.0, 5.0))

        # ── Actor ─────────────────────────────────────────────────────────
        d_actor  = -(a - mu) / (std ** 2 + 1e-8)
        d_actor *= (1 - np.tanh(h @ self.W_actor + self.b_actor) ** 2) * self.max_speed
        d_actor  = self._safe(d_actor)
        d_log_std = ((a - mu) ** 2 / (std ** 2 + 1e-8) - 1.0) - 2.0 / (std + 1e-8)

        self.W_actor += self.lr * self._clip_g(np.outer(h, d_actor))
        self.b_actor += self.lr * self._clip_g(d_actor)
        self.log_std += self.lr * 0.5 * self._clip_g(d_log_std)
        self.log_std  = np.clip(self.log_std, -3.0, 1.5)

        # ── Critics ───────────────────────────────────────────────────────
        # Use descent-consistent TD errors so critic minimizes squared TD loss.
        td_err1 = td_target - v1
        td_err2 = td_target - v2
        d_c1 = 2.0 * td_err1
        d_c2 = 2.0 * td_err2
        self.W_critic  += self.lr * self._clip_g(d_c1 * h.reshape(-1, 1))
        self.b_critic  += self.lr * self._clip_g(np.array([d_c1]))
        self.W_critic2 += self.lr * self._clip_g(d_c2 * h.reshape(-1, 1))
        self.b_critic2 += self.lr * self._clip_g(np.array([d_c2]))

        # ── Shared base ───────────────────────────────────────────────────
        d_h = self._safe(
            d_actor @ self.W_actor.T
            + d_c1 * self.W_critic.T.flatten()
            + d_c2 * self.W_critic2.T.flatten()
        )
        d_h_relu = d_h * (h > 0)
        self.W1 += self.lr * self._clip_g(np.outer(s, d_h_relu))
        self.b1 += self.lr * self._clip_g(d_h_relu)

        return float(err1 ** 2 + err2 ** 2 - log_prob)

    def update(self) -> float:
        use_offpolicy = len(self.replay_buffer) >= max(self.batch_size, 2)
        if (not use_offpolicy) and len(self.rewards) < max(2, self.pg_update_interval):
            return 0.0

        self.alpha = float(np.clip(np.exp(self.log_alpha), 0.01, 1.0))

        # On-policy Monte Carlo targets
        G = 0.0
        targets = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            targets.insert(0, G)
        targets = np.array(targets, dtype=np.float64)

        # FIX 9
        if len(targets) > 1 and targets.std() > 1e-6:
            targets = (targets - targets.mean()) / (targets.std() + 1e-8)
        targets = np.clip(targets, -5.0, 5.0)

        total_loss    = 0.0
        total_entropy = 0.0

        if use_offpolicy:
            idx   = np.random.choice(len(self.replay_buffer),
                                     size=min(self.batch_size, len(self.replay_buffer)),
                                     replace=False)
            batch = [self.replay_buffer[i] for i in idx]
            for s, a, r in batch:
                # Twin-critic TD target from target networks
                h_t  = self._forward_base(s, target=True)
                vt1  = self._forward_critic(h_t, target=True)
                vt2  = self._forward_critic2(h_t, target=True)
                mu_t, std_t = self._forward_actor(h_t)
                lp_t = -0.5 * np.sum(((a - mu_t) / (std_t + 1e-8)) ** 2) \
                       - np.sum(np.log(std_t + 1e-8))
                ent_t = -lp_t
                # FIX 14: clip td_target before passing into update
                td_target = float(np.clip(
                    r + self.gamma * (min(vt1, vt2) + self.alpha * ent_t),
                    -10.0, 10.0))
                total_loss    += self._update_one(s, a, r, td_target)
                total_entropy += ent_t
        else:
            for s, a, target in zip(self.states, self.actions, targets):
                h_t  = self._forward_base(s, target=True)
                vt1  = self._forward_critic(h_t, target=True)
                vt2  = self._forward_critic2(h_t, target=True)
                mu_t, std_t = self._forward_actor(h_t)
                lp_t = -0.5 * np.sum(((a - mu_t) / (std_t + 1e-8)) ** 2) \
                       - np.sum(np.log(std_t + 1e-8))
                ent_t = -lp_t
                td_target = float(np.clip(
                    target + self.alpha * ent_t,
                    -10.0, 10.0))
                total_loss    += self._update_one(s, a, target, td_target)
                total_entropy += ent_t

        # Temperature update
        n = max(len(self.states) if not use_offpolicy else self.batch_size, 1)
        mean_entropy = total_entropy / n
        alpha_loss   = -self.log_alpha * (mean_entropy + self.target_entropy)
        self.log_alpha = float(np.clip(
            self.log_alpha - self.alpha_lr * alpha_loss, -5.0, 2.0))

        # Soft target update
        τ = self.tau
        self.W_critic_target  = (1 - τ) * self.W_critic_target  + τ * self.W_critic
        self.b_critic_target  = (1 - τ) * self.b_critic_target  + τ * self.b_critic
        self.W_critic2_target = (1 - τ) * self.W_critic2_target + τ * self.W_critic2
        self.b_critic2_target = (1 - τ) * self.b_critic2_target + τ * self.b_critic2
        self.W1_target = (1 - τ) * self.W1_target + τ * self.W1
        self.b1_target = (1 - τ) * self.b1_target + τ * self.b1

        self.episode_returns.append(float(np.sum(self.rewards)))
        self.policy_loss_history.append(total_loss / n)

        # Push on-policy transitions into replay buffer
        for s, a, r in zip(self.states, self.actions, self.rewards):
            if len(self.replay_buffer) >= self.replay_size:
                self.replay_buffer.pop(0)
            self.replay_buffer.append((s, a, r))

        self._clip_weights()
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        return total_loss / n


# ============================================================
# FACTORY
# ============================================================

def create_policy(method: str, **kwargs) -> PolicyNetworkBase:
    methods = {
        'reinforce': PolicyNetworkREINFORCE,
        'a2c':       PolicyNetworkA2C,
        'soft_a2c':  PolicyNetworkSoftA2C,
    }
    if method.lower() not in methods:
        raise ValueError(
            f"Unknown policy method: {method}. Available: {list(methods.keys())}")
    return methods[method.lower()](**kwargs)