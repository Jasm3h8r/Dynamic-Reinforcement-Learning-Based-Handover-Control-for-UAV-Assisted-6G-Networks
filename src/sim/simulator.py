import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyArrow
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from .policy_gradient import create_policy, PolicyNetworkBase

# Custom Gaussian smoothing filter
def gaussian_filter(array, sigma=3):
    size = int(sigma * 4)
    x = np.arange(-size, size + 1)
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    result = array.copy()
    for i in range(result.shape[0]):
        result[i, :] = np.convolve(result[i, :], kernel_1d, mode='same')
    for j in range(result.shape[1]):
        result[:, j] = np.convolve(result[:, j], kernel_1d, mode='same')
    return result

# 6G FREQUENCY BANDS
BAND_6G = {
    'sub6':       {'freq': 3.5,  'bw': 100},
    'mmwave_low': {'freq': 28,   'bw': 400},
    'mmwave_high':{'freq': 47,   'bw': 800},
    'subthz_low': {'freq': 95,   'bw': 2000},
    'subthz_mid': {'freq': 140,  'bw': 5000}
}

MAX_SPECTRAL_EFFICIENCY_6G = 12.0
MIN_SINR_DB = -10.0
MAX_SINR_DB  = 35.0

SUBTHZ_ABSORPTION = {95: 0.5, 140: 2.0}

# ============================================================
# DRONE BATTERY / ENERGY CONSTANTS
# ============================================================
BATTERY_CAPACITY_WH      = 500.0   # Wh  â€” full charge capacity
BATTERY_CRITICAL_PCT     = 0.10    # 10% â†’ forced landing / charge
BATTERY_RESUME_PCT       = 0.90    # 90% â†’ cleared to launch again
HOVER_POWER_W            = 800.0   # W   â€” power just to stay aloft
MOVE_POWER_COEFF         = 3.0     # W per (m/step) of horizontal speed
VERTICAL_POWER_COEFF     = 5.0     # W per (m/step) of vertical speed
TX_POWER_OVERHEAD_W      = 15.0    # W   â€” RF electronics while active
CHARGE_RATE_W            = 1200.0  # W   â€” ground charging station rate
STEP_DURATION_S          = 1.0     # seconds per simulation step

# DroneState enum-like constants
DRONE_ACTIVE   = 'active'    # airborne, serving UEs
DRONE_CHARGING = 'charging'  # grounded, battery recharging
DRONE_RTB      = 'rtb'       # returning to base (low battery)

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class BaseStation6G:
    id: int
    x: float
    y: float
    z: float
    freq: float
    power: float
    beam_direction: float
    elevation_angle: float
    bs_type: str  # 'macro', 'micro', 'pico', 'drone'

    num_antennas: int = 256
    antenna_spacing: float = 0.5
    max_simultaneous_beams: int = 64
    beamforming_gain_db: float = 30.0
    bandwidth_mhz: float = 2000.0

    load: int = 0
    active_beams: Dict[int, int] = field(default_factory=dict)
    beam_angles: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # Drone-specific
    is_drone: bool = False
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    policy: Optional[object] = field(default=None, repr=False)
    handovers_caused: int = 0
    coverage_history: List[float] = field(default_factory=list)

    # â”€â”€ Battery / On-Off system â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    drone_state: str = DRONE_ACTIVE          # 'active' | 'charging' | 'rtb'
    battery_wh: float = BATTERY_CAPACITY_WH  # current charge (Wh)
    battery_pct: float = 1.0                 # 0.0 â€“ 1.0 convenience
    base_x: float = 0.0                      # charging station X
    base_y: float = 0.0                      # charging station Y

    # Lifecycle counters
    charge_cycles: int = 0          # how many times drone has charged
    total_active_steps: int = 0     # steps spent serving UEs
    total_charging_steps: int = 0   # steps spent charging
    total_rtb_steps: int = 0        # steps spent flying home
    energy_consumed_wh: float = 0.0 # cumulative energy used while active
    energy_charged_wh: float = 0.0  # cumulative energy recharged

    # Per-step energy bookkeeping (set each step for reward access)
    last_energy_draw_w: float = 0.0  # instantaneous power this step


@dataclass
class UserEquipment6G:
    id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float = 0.0
    num_antennas: int = 8
    connected_bs: Optional[int] = None
    beam_id: Optional[int] = None
    sinr: float = 0.0
    throughput: float = 0.0
    candidate_bs: Optional[int] = None
    ttt_counter: int = 0
    ttt_threshold: int = 5
    handover_count: int = 0
    time_since_last_handover: int = 0
    outage_count: int = 0


# ============================================================
# CONTEXTUAL LinUCB
# ============================================================

class ContextualLinUCB:
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A: Dict = {}
        self.b: Dict = {}
        self.total_reward = 0.0
        self.optimal_reward = 0.0
        self.decision_count = 0

    def _init_ue(self, ue_id):
        if ue_id not in self.A:
            self.A[ue_id] = [np.identity(self.n_features) for _ in range(self.n_arms)]
            self.b[ue_id] = [np.zeros(self.n_features) for _ in range(self.n_arms)]

    def select_arm(self, ue_id, contexts, current_arm=None,
                   handover_penalty=0.35):
        self._init_ue(ue_id)
        best_arm, best_ucb = 0, -np.inf
        ucb_values = []
        for arm in range(self.n_arms):
            ctx = contexts[arm]
            try:
                A_inv = np.linalg.inv(self.A[ue_id][arm])
                theta = A_inv @ self.b[ue_id][arm]
                exp_r = theta @ ctx
                if current_arm is not None and arm != current_arm:
                    exp_r -= handover_penalty
                unc = self.alpha * np.sqrt(ctx @ A_inv @ ctx)
                ucb = exp_r + unc
                ucb_values.append({'arm': arm, 'ucb': ucb,
                                   'expected_reward': exp_r, 'uncertainty': unc})
                if ucb > best_ucb:
                    best_ucb, best_arm = ucb, arm
            except np.linalg.LinAlgError:
                ucb_values.append({'arm': arm, 'ucb': 0,
                                   'expected_reward': 0, 'uncertainty': 0})
        return best_arm, ucb_values, (current_arm is None or best_arm != current_arm)

    def update(self, ue_id, arm, ctx, reward):
        self._init_ue(ue_id)
        self.A[ue_id][arm] += np.outer(ctx, ctx)
        self.b[ue_id][arm] += reward * ctx
        self.decision_count += 1

    def add_arm(self):
        """Add a new arm (for newly added drone BS)"""
        self.n_arms += 1
        for ue_id in self.A:
            self.A[ue_id].append(np.identity(self.n_features))
            self.b[ue_id].append(np.zeros(self.n_features))

    def reset(self):
        self.A, self.b = {}, {}
        self.total_reward = self.optimal_reward = 0.0
        self.decision_count = 0


# ============================================================
# MAIN SIMULATOR
# ============================================================

class HetNet6GSimulator:
    """6G Ground HetNet + Drone BS with Policy Gradient RL"""

    def __init__(self,
                 grid_size: int = 3000,
                 num_macro: int = 4, num_micro: int = 8, num_pico: int = 12,
                 num_drones: int = 6,
                 num_ues: int = 300,
                 tx_power_macro: float = 43,
                 tx_power_micro: float = 35,
                 tx_power_pico: float = 28,
                 tx_power_drone: float = 38,
                 noise_power: float = -174,
                 linucb_alpha: float = 1.0,
                 drone_pg_lr: float = 3e-3,
                 drone_max_speed: float = 15.0,
                 drone_z_min: float = 30.0,
                 drone_z_max: float = 150.0,
                 drone_update_interval: int = 10,
                 policy_method: str = 'reinforce',
                 seed: Optional[int] = None):

        if seed is not None:
            np.random.seed(seed)

        self.grid_size   = grid_size
        self.num_macro   = num_macro
        self.num_micro   = num_micro
        self.num_pico    = num_pico
        self.num_drones  = num_drones
        self.num_static_bs = num_macro + num_micro + num_pico
        self.num_bs      = self.num_static_bs + num_drones
        self.num_ues     = num_ues
        self.tx_power_macro = tx_power_macro
        self.tx_power_micro = tx_power_micro
        self.tx_power_pico = tx_power_pico
        self.tx_power_drone = tx_power_drone
        self.noise_power = noise_power
        self.linucb_alpha = linucb_alpha
        self.drone_pg_lr = drone_pg_lr
        self.drone_max_speed = drone_max_speed
        self.drone_z_min = drone_z_min
        self.drone_z_max = drone_z_max
        self.drone_update_interval = drone_update_interval
        self.policy_method = policy_method
        self.seed = seed

        self.base_stations: List[BaseStation6G] = []
        self.users:         List[UserEquipment6G] = []
        self.cmab:          ContextualLinUCB = None

        self.channel_cache    = {}
        self.shadow_fading_map= {}

        self.step = 0
        self.logs = []
        self.step_metrics = {
            'avg_sinr': [], 'total_throughput': [], 'handover_count': [],
            'coverage': [], 'regret': [], 'avg_load': [],
            'cell_edge_rate': [], 'outage_prob': [], 'pingpong_rate': [],
            'drone_handover_reduction': [], 'drone_positions': [],
            'drone_rewards': [],
            # battery metrics
            'avg_battery_pct': [],       # mean SoC across all drones
            'drones_active': [],          # count of ACTIVE drones
            'drones_charging': [],        # count of CHARGING drones
            'drones_rtb': [],             # count of RTB drones
            'total_energy_consumed_wh': [], # cumulative energy this step
            'fleet_efficiency': [],       # Mbps per watt fleet-wide
        }
        self.throughput_distribution = []
        self.sinr_samples = []

        # Baseline handover rate (before drone learning kicks in)
        self._baseline_ho_rate: Optional[float] = None

        self._initialize_hetnet()
        self._initialize_shadow_map()

    # ------------------------------------------------------------------ #
    #  INITIALIZATION                                                       #
    # ------------------------------------------------------------------ #

    def _initialize_hetnet(self):
        self.base_stations = self._deploy_6g_hetnet()
        self.users         = self._generate_users()
        self.cmab          = ContextualLinUCB(
            n_arms=self.num_bs, n_features=10, alpha=self.linucb_alpha)
        self._log(f'6G+Drone HetNet: {self.num_macro}M+{self.num_micro}m+'
                  f'{self.num_pico}p+{self.num_drones}D BSs, {self.num_ues} UEs')

    def _initialize_shadow_map(self):
        gp = 50
        x  = np.linspace(0, self.grid_size, gp)
        y  = np.linspace(0, self.grid_size, gp)
        for bs in self.base_stations:
            sigma = 8.0 if bs.freq > 90 else 4.0
            sm = np.random.normal(0, sigma, (gp, gp))
            sm = gaussian_filter(sm, sigma=3)
            self.shadow_fading_map[bs.id] = {'x': x, 'y': y, 'values': sm}

    def _deploy_6g_hetnet(self) -> List[BaseStation6G]:
        bs_list = []
        bid = 0

        # ---- Macro ----
        for i in range(self.num_macro):
            angle = (2 * np.pi * i) / self.num_macro
            r = self.grid_size / 3.5
            x = self.grid_size/2 + r * np.cos(angle)
            y = self.grid_size/2 + r * np.sin(angle)
            freq = np.random.choice([3.5, 28])
            bw   = 100 if freq < 10 else 400
            bs_list.append(BaseStation6G(
                id=bid, x=x, y=y, z=30.0, freq=freq,
                power=self.tx_power_macro,
                beam_direction=np.random.uniform(0,360),
                elevation_angle=np.random.uniform(-10,10),
                bs_type='macro', num_antennas=128,
                max_simultaneous_beams=256,
                beamforming_gain_db=25.0, bandwidth_mhz=bw))
            bid += 1

        # ---- Micro ----
        for i in range(self.num_micro):
            x = np.random.uniform(self.grid_size*.15, self.grid_size*.85)
            y = np.random.uniform(self.grid_size*.15, self.grid_size*.85)
            freq = np.random.choice([28, 47, 95])
            bw   = 400 if freq < 50 else 2000
            bs_list.append(BaseStation6G(
                id=bid, x=x, y=y, z=15.0, freq=freq,
                power=self.tx_power_micro,
                beam_direction=np.random.uniform(0,360),
                elevation_angle=np.random.uniform(-15,15),
                bs_type='micro', num_antennas=256,
                max_simultaneous_beams=128,
                beamforming_gain_db=30.0, bandwidth_mhz=bw))
            bid += 1

        # ---- Pico ----
        for i in range(self.num_pico):
            x = np.random.uniform(0, self.grid_size)
            y = np.random.uniform(0, self.grid_size)
            freq = np.random.choice([95, 140])
            bw   = 2000 if freq < 120 else 5000
            bs_list.append(BaseStation6G(
                id=bid, x=x, y=y, z=8.0, freq=freq,
                power=self.tx_power_pico,
                beam_direction=np.random.uniform(0,360),
                elevation_angle=np.random.uniform(-20,20),
                bs_type='pico', num_antennas=512,
                max_simultaneous_beams=64,
                beamforming_gain_db=35.0, bandwidth_mhz=bw))
            bid += 1

        # ---- Drone BS (new!) ----
        for i in range(self.num_drones):
            x = np.random.uniform(self.grid_size*.1, self.grid_size*.9)
            y = np.random.uniform(self.grid_size*.1, self.grid_size*.9)
            z = np.random.uniform(self.drone_z_min, self.drone_z_max)
            freq = np.random.choice([28, 47, 95])
            bw   = 400 if freq < 50 else 2000

            policy = create_policy(
                self.policy_method,
                state_dim=13, action_dim=3,
                lr=self.drone_pg_lr,
                gamma=0.97,
                max_speed=self.drone_max_speed,
                z_min=self.drone_z_min,
                z_max=self.drone_z_max
            )

            bs_list.append(BaseStation6G(
                id=bid, x=x, y=y, z=z, freq=freq,
                power=self.tx_power_drone,
                beam_direction=np.random.uniform(0,360),
                elevation_angle=np.random.uniform(-30,30),
                bs_type='drone',
                num_antennas=64,
                max_simultaneous_beams=32,
                beamforming_gain_db=20.0,
                bandwidth_mhz=bw,
                is_drone=True,
                policy=policy,
                base_x=x,    # charging station = initial deploy position
                base_y=y,
                battery_wh=BATTERY_CAPACITY_WH * np.random.uniform(0.6, 1.0),
            ))
            bid += 1

        return bs_list

    def _generate_users(self) -> List[UserEquipment6G]:
        return [UserEquipment6G(
            id=i,
            x=np.random.uniform(0, self.grid_size),
            y=np.random.uniform(0, self.grid_size),
            z=1.5,
            vx=np.random.uniform(-2, 2),
            vy=np.random.uniform(-2, 2),
            num_antennas=np.random.choice([4,8,16])
        ) for i in range(self.num_ues)]

    # ------------------------------------------------------------------ #
    #  DRONE STATE / REWARD                                                 #
    # ------------------------------------------------------------------ #

    def _get_drone_state(self, drone: BaseStation6G) -> np.ndarray:
        """Build normalized state vector for drone policy (13 features)."""
        connected_ues = [u for u in self.users if u.connected_bs == drone.id]
        n_conn = len(connected_ues)

        if connected_ues:
            avg_ue_x = np.mean([u.x for u in connected_ues]) / self.grid_size
            avg_ue_y = np.mean([u.y for u in connected_ues]) / self.grid_size
            coverage_score = n_conn / max(self.num_ues / self.num_bs, 1)
        else:
            avg_ue_x = drone.x / self.grid_size
            avg_ue_y = drone.y / self.grid_size
            coverage_score = 0.0

        load_ratio = n_conn / drone.max_simultaneous_beams

        other_bs = [bs for bs in self.base_stations if bs.id != drone.id]
        dists = sorted([
            self._dist3d(drone.x, drone.y, drone.z, bs.x, bs.y, bs.z)
            for bs in other_bs
        ])[:3]
        while len(dists) < 3:
            dists.append(self.grid_size)
        dists_norm = [d / self.grid_size for d in dists]

        ho_rate = 0.0
        if len(self.step_metrics['handover_count']) > 0:
            recent = self.step_metrics['handover_count'][-min(10, len(self.step_metrics['handover_count'])):]
            ho_rate = np.mean(recent) / max(self.num_ues, 1)

        outage_rate = 0.0
        if len(self.step_metrics['outage_prob']) > 0:
            outage_rate = self.step_metrics['outage_prob'][-1] / 100.0

        # Distance to own charging base (normalised)
        dist_to_base = np.sqrt((drone.x - drone.base_x)**2 +
                               (drone.y - drone.base_y)**2) / self.grid_size

        state = np.array([
            drone.x / self.grid_size,
            drone.y / self.grid_size,
            (drone.z - self.drone_z_min) / (self.drone_z_max - self.drone_z_min),
            avg_ue_x,
            avg_ue_y,
            np.clip(load_ratio, 0, 1),
            np.clip(coverage_score, 0, 2) / 2.0,
            dists_norm[0], dists_norm[1], dists_norm[2],
            np.clip(ho_rate, 0, 1),
            # â”€â”€ battery features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            np.clip(drone.battery_pct, 0.0, 1.0),   # [11] state-of-charge
            np.clip(dist_to_base, 0.0, 1.0),         # [12] dist to charger
        ], dtype=np.float32)

        return state

    def _compute_drone_reward(self, drone: BaseStation6G,
                               prev_ho_count: int, curr_ho_count: int,
                               n_connected_ues: int) -> float:
        """
        Performativity reward integrating battery state:

        ACTIVE state:
          + coverage quality (SINR-weighted)
          - handover penalty
          + gap-fill bonus
          - energy waste penalty (high power draw for little coverage)
          - low-battery urgency penalty (exponential as SoC drops)

        CHARGING state:
          + charge_progress bonus  (incentivise fast full recharge)
          - coverage_loss penalty  (network suffers while drone is down)

        RTB state:
          - per-step penalty  (time spent returning is dead time)
          + efficiency bonus  (RTB at higher altitude â†’ faster glide)
        """
        soc = drone.battery_pct   # 0.0 â€“ 1.0

        # â”€â”€ CHARGING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if drone.drone_state == DRONE_CHARGING:
            # Reward for charging progress (scaled by how empty it was)
            charge_progress = min(drone.last_energy_draw_w * STEP_DURATION_S
                                  / 3600.0 / BATTERY_CAPACITY_WH, 0.1)
            # Penalty: every step offline, nearby UEs may suffer
            nearby_struggling = sum(
                1 for u in self.users
                if self._dist3d(u.x, u.y, u.z,
                                drone.base_x, drone.base_y, drone.z) <
                   self._coverage_radius(drone) and u.sinr < 5.0
            ) / max(self.num_ues, 1)
            coverage_loss_pen = -0.3 * nearby_struggling
            return charge_progress + coverage_loss_pen

        # â”€â”€ RETURNING TO BASE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if drone.drone_state == DRONE_RTB:
            dist_remaining = np.sqrt((drone.x - drone.base_x)**2 +
                                     (drone.y - drone.base_y)**2)
            # Small per-step penalty; less penalty if making good progress
            progress_bonus = max(0, 0.05 * (1 - dist_remaining / self.grid_size))
            return -0.15 + progress_bonus

        # â”€â”€ ACTIVE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        connected_ues = [u for u in self.users if u.connected_bs == drone.id]

        # 1. SINR-weighted coverage quality
        if connected_ues:
            sinr_vals = np.array([u.sinr for u in connected_ues])
            avg_sinr  = float(np.mean(sinr_vals))
            sinr_norm = (avg_sinr - MIN_SINR_DB) / (MAX_SINR_DB - MIN_SINR_DB)
            coverage_r = np.clip(sinr_norm, 0, 1) * len(connected_ues) / \
                         max(self.num_ues / self.num_bs, 1)
        else:
            coverage_r = -0.05   # hovering idle is wasteful

        # 2. Handover penalty (drone movement induced)
        ho_delta   = max(curr_ho_count - prev_ho_count, 0)
        ho_pen     = -0.08 * ho_delta

        # 3. Gap-fill bonus: reward for serving UEs with no strong alternative
        gap_bonus = 0.0
        for u in connected_ues:
            best_alt = max(
                (self._sinr(u, bs) for bs in self.base_stations
                 if bs.id != drone.id and not bs.is_drone),
                default=MIN_SINR_DB
            )
            if best_alt < 5.0:   # UE really needs this drone
                gap_bonus += 0.15 / max(self.num_ues, 1)

        # 4. Energy efficiency: penalise burning power for little return
        #    efficiency = bits_served / watts_consumed
        power_w       = max(drone.last_energy_draw_w, 1.0)
        bits_proxy    = sum(u.throughput for u in connected_ues)   # Mbps sum
        energy_eff    = bits_proxy / power_w                        # Mbps/W
        # Normalise: target ~5 Mbps/W as "good"
        eff_reward    = np.clip(energy_eff / 5.0, -0.1, 0.2) * 0.1

        # 5. Low-battery urgency penalty (exponential decay)
        #    As SoC drops below 30%, penalise staying active
        if soc < 0.30:
            urgency_pen = -0.5 * np.exp(-10.0 * soc)   # steep near 0
        else:
            urgency_pen = 0.0

        # 6. Boundary penalty (graduated)
        margin = self.grid_size * 0.05
        bx = min(drone.x, self.grid_size - drone.x)
        by = min(drone.y, self.grid_size - drone.y)
        bz = min(drone.z - self.drone_z_min, self.drone_z_max - drone.z)
        boundary_pen = 0.0
        for margin_val in [bx, by]:
            if margin_val < margin:
                boundary_pen -= 0.15 * (1 - margin_val / margin) ** 2
        if bz < 10:
            boundary_pen -= 0.1 * (1 - bz / 10) ** 2

        return (coverage_r + ho_pen + gap_bonus +
                eff_reward + urgency_pen + boundary_pen)

    # ------------------------------------------------------------------ #
    #  BATTERY STATE MACHINE                                               #
    # ------------------------------------------------------------------ #

    def _tick_battery_active(self, drone: BaseStation6G, action: np.ndarray):
        """Consume energy while drone is airborne & serving. Returns watts used."""
        speed_h = np.sqrt(drone.vx**2 + drone.vy**2)
        speed_v = abs(drone.vz)
        power_w = (HOVER_POWER_W
                   + MOVE_POWER_COEFF   * speed_h
                   + VERTICAL_POWER_COEFF * speed_v
                   + TX_POWER_OVERHEAD_W)
        energy_wh = power_w * STEP_DURATION_S / 3600.0

        drone.battery_wh       = max(drone.battery_wh - energy_wh, 0.0)
        drone.battery_pct      = drone.battery_wh / BATTERY_CAPACITY_WH
        drone.energy_consumed_wh += energy_wh
        drone.last_energy_draw_w  = power_w
        drone.total_active_steps += 1
        return power_w

    def _tick_battery_charging(self, drone: BaseStation6G):
        """Add charge while drone is at base. Returns watts added."""
        can_add_wh   = (BATTERY_CAPACITY_WH - drone.battery_wh)
        added_wh     = min(CHARGE_RATE_W * STEP_DURATION_S / 3600.0, can_add_wh)
        drone.battery_wh      += added_wh
        drone.battery_pct      = drone.battery_wh / BATTERY_CAPACITY_WH
        drone.energy_charged_wh += added_wh
        drone.last_energy_draw_w = -CHARGE_RATE_W   # negative = charging in
        drone.total_charging_steps += 1
        return CHARGE_RATE_W

    def _tick_rtb(self, drone: BaseStation6G):
        """Move drone toward its charging base. Returns True when arrived."""
        dx = drone.base_x - drone.x
        dy = drone.base_y - drone.y
        dist = np.sqrt(dx**2 + dy**2)

        rtb_speed = min(self.drone_max_speed * 1.5, dist)  # faster when RTB
        if dist > 0.5:
            drone.vx = (dx / dist) * rtb_speed
            drone.vy = (dy / dist) * rtb_speed
            drone.x  = np.clip(drone.x + drone.vx, 0, self.grid_size)
            drone.y  = np.clip(drone.y + drone.vy, 0, self.grid_size)
            # Descend toward z_min during RTB
            if drone.z > self.drone_z_min + 5:
                drone.vz = -3.0
                drone.z  = max(drone.z + drone.vz, self.drone_z_min)

        # Consume energy while flying home (no TX overhead)
        speed_h = np.sqrt(drone.vx**2 + drone.vy**2)
        power_w = HOVER_POWER_W + MOVE_POWER_COEFF * speed_h
        energy_wh = power_w * STEP_DURATION_S / 3600.0
        drone.battery_wh       = max(drone.battery_wh - energy_wh, 0.0)
        drone.battery_pct      = drone.battery_wh / BATTERY_CAPACITY_WH
        drone.energy_consumed_wh += energy_wh
        drone.last_energy_draw_w  = power_w
        drone.total_rtb_steps    += 1

        arrived = dist < 1.0
        if arrived:
            drone.x = drone.base_x
            drone.y = drone.base_y
            drone.z = self.drone_z_min
            drone.vx = drone.vy = drone.vz = 0.0
        return arrived

    def _transition_drone_state(self, drone: BaseStation6G):
        """
        State machine transitions:
          active   â†’ rtb       : battery â‰¤ CRITICAL  (forced)
          rtb      â†’ charging  : arrived at base
          charging â†’ active    : battery â‰¥ RESUME
        """
        if drone.drone_state == DRONE_ACTIVE:
            if drone.battery_pct <= BATTERY_CRITICAL_PCT:
                drone.drone_state = DRONE_RTB
                self._log(f'Drone {drone.id} â†’ RTB (battery {drone.battery_pct*100:.1f}%)')
                # Force-handover all connected UEs
                for u in self.users:
                    if u.connected_bs == drone.id:
                        u.connected_bs  = None
                        u.candidate_bs  = None
                        u.ttt_counter   = 0

        elif drone.drone_state == DRONE_RTB:
            arrived = self._tick_rtb(drone)
            if arrived:
                drone.drone_state  = DRONE_CHARGING
                drone.charge_cycles += 1
                self._log(f'Drone {drone.id} â†’ CHARGING (cycle #{drone.charge_cycles})')

        elif drone.drone_state == DRONE_CHARGING:
            self._tick_battery_charging(drone)
            if drone.battery_pct >= BATTERY_RESUME_PCT:
                drone.drone_state = DRONE_ACTIVE
                # Relaunch: climb to operating altitude
                drone.z  = self.drone_z_min
                drone.vz = 5.0   # climb at 5 m/step
                self._log(f'Drone {drone.id} â†’ ACTIVE '
                          f'(battery {drone.battery_pct*100:.1f}%)')

    # ------------------------------------------------------------------ #
    #  DRONE MOVEMENT (replaces old _move_drones)                          #
    # ------------------------------------------------------------------ #

    def _move_drones(self, prev_ho_count: int, curr_ho_count: int):
        """Battery-aware drone step: state transitions + policy decisions."""
        for bs in self.base_stations:
            if not bs.is_drone or bs.policy is None:
                continue

            # â”€â”€ State transitions first â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if bs.drone_state != DRONE_ACTIVE:
                self._transition_drone_state(bs)
                # Store a "null" experience during off-time
                state  = self._get_drone_state(bs)
                reward = self._compute_drone_reward(
                    bs, prev_ho_count, curr_ho_count, 0)
                bs.policy.store(state, np.zeros(3), reward)
                if self.step % self.drone_update_interval == 0:
                    bs.policy.update()
                continue   # skip movement logic while not active

            # â”€â”€ ACTIVE: policy decides movement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Check if we should initiate RTB proactively
            # (drone learns to head home before forced at critical level)
            state  = self._get_drone_state(bs)   # includes battery_pct
            action = bs.policy.select_action(state)

            # If battery is critically low, override policy â†’ force RTB
            if bs.battery_pct <= BATTERY_CRITICAL_PCT:
                self._transition_drone_state(bs)  # â†’ RTB
                continue

            # Apply movement
            new_x = np.clip(bs.x + action[0], 0, self.grid_size)
            new_y = np.clip(bs.y + action[1], 0, self.grid_size)
            new_z = np.clip(bs.z + action[2],
                            self.drone_z_min, self.drone_z_max)

            bs.vx = new_x - bs.x
            bs.vy = new_y - bs.y
            bs.vz = new_z - bs.z
            bs.x, bs.y, bs.z = new_x, new_y, new_z

            # â”€â”€ Energy tick (after movement so speed is updated) â”€â”€â”€â”€â”€â”€â”€â”€â”€
            self._tick_battery_active(bs, action)

            # Reward (battery state baked into _compute_drone_reward)
            n_conn  = sum(1 for u in self.users if u.connected_bs == bs.id)
            reward  = self._compute_drone_reward(
                bs, prev_ho_count, curr_ho_count, n_conn)

            bs.policy.store(state, action, reward)

            # Periodic PG update
            if self.step % self.drone_update_interval == 0:
                bs.policy.update()

            # Transition check after energy draw
            if bs.battery_pct <= BATTERY_CRITICAL_PCT:
                self._transition_drone_state(bs)

    # ------------------------------------------------------------------ #
    #  CHANNEL / PATH LOSS                                                  #
    # ------------------------------------------------------------------ #

    def _log(self, msg):
        self.logs.append(f'[S{self.step}] {msg}')
        if len(self.logs) > 30:
            self.logs = self.logs[-30:]

    @staticmethod
    def _dist3d(x1,y1,z1,x2,y2,z2):
        return np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

    @staticmethod
    def _azimuth(x1,y1,x2,y2):
        return np.arctan2(y2-y1,x2-x1)*180/np.pi

    @staticmethod
    def _elevation(x1,y1,z1,x2,y2,z2):
        hd = np.sqrt((x1-x2)**2+(y1-y2)**2)
        return 0.0 if hd < 0.1 else np.arctan2(z2-z1,hd)*180/np.pi

    @staticmethod
    def _los_prob(d, freq, bs_type):
        if freq > 90:   d0,d1 = 100, 40
        elif bs_type=='macro': d0,d1 = 250,150
        elif bs_type=='micro': d0,d1 = 150, 80
        else:                  d0,d1 = 80,  40
        return min(d0/max(d,1),1)*(1-np.exp(-d/d1))+np.exp(-d/d1)

    def _mol_absorption(self, freq, dist_m):
        if freq < 90: return 0.0
        abs_db_km = next((v for f,v in SUBTHZ_ABSORPTION.items()
                          if abs(freq-f)<50), 0.0)
        return abs_db_km * dist_m / 1000.0

    def _path_loss(self, d, fc, is_los, z_bs, z_ue):
        d = max(d, 1)
        if is_los:
            pl = 32.4 + 21*np.log10(d) + 20*np.log10(fc)
        else:
            pl = 32.4 + 21*np.log10(d) + 20*np.log10(fc) + 25 + 0.6*(fc-24)
        if fc > 90:
            pl += self._mol_absorption(fc, d)
        hd = abs(z_bs - z_ue)
        if hd > 10:
            pl += 2*np.log10(hd/10)
        # Drone altitude advantage (LOS probability higher â†’ slight gain)
        if z_bs > 50:
            pl -= 2.0
        return pl

    @staticmethod
    def _rayleigh():
        r,i = np.random.normal(0,1), np.random.normal(0,1)
        return np.clip(20*np.log10(np.sqrt(r**2+i**2)/np.sqrt(2)), -10, 6)

    @staticmethod
    def _rician(k_db=12.0):
        k = 10**(k_db/10)
        los = np.sqrt(k/(k+1))
        sp  = 1/(k+1)
        r   = np.random.normal(0,np.sqrt(sp/2))
        i   = np.random.normal(0,np.sqrt(sp/2))
        return np.clip(20*np.log10(np.sqrt((los+r)**2+i**2)), -5, 10)

    def _get_or_create_channel(self, user, bs):
        key = (user.id, bs.id)
        if key not in self.channel_cache:
            d = self._dist3d(user.x,user.y,user.z,bs.x,bs.y,bs.z)
            is_los = np.random.uniform() < self._los_prob(d,bs.freq,bs.bs_type)
            shadow = self._get_shadow(bs.id, user.x, user.y)
            if is_los:
                kf = max(8, 20-d/50)
                small = self._rician(kf)
            else:
                small = self._rayleigh()
            self.channel_cache[key] = {'los':is_los,'shadow':shadow,'small_scale':small}
        return self.channel_cache[key]

    def _get_shadow(self, bs_id, ux, uy):
        sm = self.shadow_fading_map[bs_id]
        xi = int((ux/self.grid_size)*(len(sm['x'])-1))
        yi = int((uy/self.grid_size)*(len(sm['y'])-1))
        xi = np.clip(xi,0,len(sm['x'])-1)
        yi = np.clip(yi,0,len(sm['y'])-1)
        return np.clip(sm['values'][yi,xi],-15,15)

    def _bf_gain(self, bs, user):
        ua = self._azimuth(bs.x,bs.y,user.x,user.y)
        ue = self._elevation(bs.x,bs.y,bs.z,user.x,user.y,user.z)
        if user.id in bs.active_beams:
            ba,be = bs.beam_angles.get(bs.active_beams[user.id],(ua,ue))
        else:
            ba,be = ua,ue
        ad = abs(((ua-ba+180)%360)-180)
        ed = abs(ue-be)
        bw = 10 if bs.freq>90 else (12 if bs.is_drone else 15)
        if ad<bw/2 and ed<bw/2:    return bs.beamforming_gain_db
        elif ad<bw  and ed<bw:     return bs.beamforming_gain_db-3
        else:
            pen = np.sqrt(ad**2+ed**2)
            return max(bs.beamforming_gain_db-20-0.15*(pen-bw), 0)

    def _freq_interfere(self, f1, f2):
        return abs(f1-f2) < 15

    def _sinr(self, user, bs, loads=None):
        d = self._dist3d(user.x,user.y,user.z,bs.x,bs.y,bs.z)
        ch = self._get_or_create_channel(user,bs)
        pl = self._path_loss(d,bs.freq,ch['los'],bs.z,user.z)
        bfg= self._bf_gain(bs,user)
        rx = bs.power + bfg - pl + ch['shadow'] + ch['small_scale']

        interference = 0
        for obs in self.base_stations:
            if obs.id==bs.id: continue
            ol = (loads.get(obs.id,0) if loads else obs.load)
            if ol==0: continue
            if not self._freq_interfere(bs.freq,obs.freq): continue
            di = self._dist3d(user.x,user.y,user.z,obs.x,obs.y,obs.z)
            cr = self._coverage_radius(obs)*1.5
            if di>cr: continue
            ci = self._get_or_create_channel(user,obs)
            pi = self._path_loss(di,obs.freq,ci['los'],obs.z,user.z)
            ir = (10 if obs.freq>90 else 5)
            interference += 10**((obs.power-pi+ci['shadow']+ci['small_scale']-ir)/10)

        noise  = 10**(self.noise_power/10)
        signal = 10**(rx/10)
        sinr   = 10*np.log10(signal/(interference+noise))
        return np.clip(sinr, MIN_SINR_DB, MAX_SINR_DB)

    def _throughput(self, sinr, bw_mhz):
        se = min(np.log2(1+10**(sinr/10)), MAX_SPECTRAL_EFFICIENCY_6G)
        return bw_mhz * se

    def _get_context(self, user, bs, loads=None):
        d    = self._dist3d(user.x,user.y,user.z,bs.x,bs.y,bs.z)
        az   = self._azimuth(bs.x,bs.y,user.x,user.y)
        el   = self._elevation(bs.x,bs.y,bs.z,user.x,user.y,user.z)
        sinr = self._sinr(user, bs, loads)
        cl   = loads.get(bs.id,0) if loads else bs.load
        return np.array([
            (sinr-MIN_SINR_DB)/(MAX_SINR_DB-MIN_SINR_DB),
            d/self.grid_size,
            (az+180)/360,
            (el+90)/180,
            bs.freq/150,
            bs.power/50,
            bs.bandwidth_mhz/5000,
            bs.num_antennas/512,
            cl/max(self.num_ues/self.num_bs*2,1),
            1.0 if bs.freq>90 else (0.5 if bs.freq>20 else 0.0)
        ])

    def _coverage_radius(self, bs, sinr_th=0.0):
        lo,hi = 1, self.grid_size
        for _ in range(20):
            td = (lo+hi)/2
            il = self._los_prob(td,bs.freq,bs.bs_type)>0.5
            pl = self._path_loss(td,bs.freq,il,bs.z,1.5)
            rx = bs.power+bs.beamforming_gain_db-pl
            est= 10*np.log10(10**(rx/10)/(10**(-100/10)+10**(self.noise_power/10)))
            if est>sinr_th: lo=td
            else: hi=td
        return lo

    def _reward(self, tput, ho, prev_tput=0, sinr=0):
        r = tput/1000.0
        if ho and (tput-prev_tput)/1000.0 < 0.025:
            r -= 0.1
        if sinr>20: r += 0.05
        return r

    # ------------------------------------------------------------------ #
    #  RESET                                                                #
    # ------------------------------------------------------------------ #

    def reset(self):
        for u in self.users:
            u.x  = np.random.uniform(0,self.grid_size)
            u.y  = np.random.uniform(0,self.grid_size)
            u.z  = 1.5
            u.vx = np.random.uniform(-2,2)
            u.vy = np.random.uniform(-2,2)
            u.connected_bs = None; u.beam_id = None
            u.sinr=0; u.throughput=0; u.handover_count=0
            u.time_since_last_handover=0; u.candidate_bs=None
            u.ttt_counter=0; u.outage_count=0

        for bs in self.base_stations:
            bs.load=0; bs.active_beams.clear(); bs.beam_angles.clear()
            bs.handovers_caused=0
            if bs.is_drone:
                bs.drone_state       = DRONE_ACTIVE
                bs.battery_wh        = BATTERY_CAPACITY_WH * np.random.uniform(0.6, 1.0)
                bs.battery_pct       = bs.battery_wh / BATTERY_CAPACITY_WH
                bs.charge_cycles     = 0
                bs.total_active_steps   = 0
                bs.total_charging_steps = 0
                bs.total_rtb_steps      = 0
                bs.energy_consumed_wh   = 0.0
                bs.energy_charged_wh    = 0.0
                bs.last_energy_draw_w   = 0.0
                bs.vx = bs.vy = bs.vz  = 0.0

        self.cmab.reset()
        self.step=0; self.channel_cache={}
        self._initialize_shadow_map()
        self.step_metrics={k:[] for k in self.step_metrics}
        self.throughput_distribution=[]; self.sinr_samples=[]
        self._baseline_ho_rate=None

    # ------------------------------------------------------------------ #
    #  SIMULATION STEP                                                      #
    # ------------------------------------------------------------------ #

    def simulation_step(self) -> Dict:
        self.channel_cache = {}

        # Move UEs
        for u in self.users:
            u.x = np.clip(u.x+u.vx, 0, self.grid_size)
            u.y = np.clip(u.y+u.vy, 0, self.grid_size)
            if u.x<=0 or u.x>=self.grid_size: u.vx=-u.vx
            if u.y<=0 or u.y>=self.grid_size: u.vy=-u.vy
            u.time_since_last_handover+=1

        prev_tputs  = {u.id: u.throughput for u in self.users}
        prev_bs_map = {u.id: u.connected_bs for u in self.users}

        for bs in self.base_stations:
            bs.load=0; bs.active_beams.clear()

        loads = {bs.id:0 for bs in self.base_stations}

        total_sinr=0; total_tput=0; ho_count=0; pingpong=0
        total_reward=0; connected=0; outage=0
        tputs=[]; sinrs=[]
        OUTAGE_TH=1.0

        for user in self.users:
            prev_bs   = user.connected_bs
            prev_tput = prev_tputs[user.id]

            # Only consider ACTIVE drones (offline drones cannot serve UEs)
            active_bs_ids = [
                bs.id for bs in self.base_stations
                if not bs.is_drone or bs.drone_state == DRONE_ACTIVE
            ]

            contexts = [self._get_context(user,bs,loads) for bs in self.base_stations]

            # If UE is connected to an offline drone, force re-selection
            if (prev_bs is not None and
                    self.base_stations[prev_bs].is_drone and
                    self.base_stations[prev_bs].drone_state != DRONE_ACTIVE):
                prev_bs = None
                user.connected_bs = None

            arm, _, should_ho = self.cmab.select_arm(
                user.id, contexts, prev_bs, handover_penalty=0.35)

            # Force arm to active BS if policy selected an offline drone
            if self.base_stations[arm].is_drone and \
                    self.base_stations[arm].drone_state != DRONE_ACTIVE:
                # Pick best active BS by context score
                arm = max(active_bs_ids,
                          key=lambda i: contexts[i][0])  # highest SINR estimate

            actual_ho = False
            final_bs  = prev_bs

            if should_ho and prev_bs is not None:
                if user.candidate_bs==arm:
                    user.ttt_counter+=1
                else:
                    user.candidate_bs=arm; user.ttt_counter=1
                if user.ttt_counter>=user.ttt_threshold:
                    actual_ho=True; final_bs=arm
                    user.ttt_counter=0; user.candidate_bs=None
                    if user.time_since_last_handover<10: pingpong+=1
                else:
                    final_bs=prev_bs
            else:
                final_bs=arm
                user.candidate_bs=None; user.ttt_counter=0

            sel_bs = self.base_stations[final_bs]

            def assign_user(bs_id, ctx):
                nonlocal total_sinr,total_tput,ho_count,pingpong,total_reward
                nonlocal connected,outage
                nonlocal actual_ho
                bss = self.base_stations[bs_id]
                sinr = self._sinr(user,bss,loads)
                tput = self._throughput(sinr,bss.bandwidth_mhz)
                if bs_id != prev_bs and prev_bs is not None:
                    was_recent_handover = user.time_since_last_handover < 10
                    actual_ho=True; ho_count+=1
                    user.handover_count+=1; user.time_since_last_handover=0
                    if was_recent_handover:
                        pingpong+=1
                    # Track if caused by drone movement
                    if bss.is_drone or (prev_bs is not None and self.base_stations[prev_bs].is_drone):
                        bss.handovers_caused+=1
                elif actual_ho:
                    ho_count+=1; user.handover_count+=1; user.time_since_last_handover=0

                rw = self._reward(tput,actual_ho,prev_tput,sinr)
                self.cmab.update(user.id,bs_id,ctx,rw)

                bid2 = loads[bss.id]
                baz  = self._azimuth(bss.x,bss.y,user.x,user.y)
                bel  = self._elevation(bss.x,bss.y,bss.z,user.x,user.y,user.z)
                bss.active_beams[user.id]=bid2
                bss.beam_angles[bid2]=(baz,bel)
                bss.load+=1; loads[bss.id]+=1

                user.connected_bs=bs_id; user.beam_id=bid2
                user.sinr=sinr; user.throughput=tput

                if tput<OUTAGE_TH: outage+=1; user.outage_count+=1
                total_sinr+=sinr; total_tput+=tput; total_reward+=rw
                connected+=1; tputs.append(tput); sinrs.append(sinr)

            if loads[sel_bs.id] < sel_bs.max_simultaneous_beams:
                assign_user(final_bs, contexts[final_bs])
            else:
                alts = sorted(
                    [(i, contexts[i],
                      contexts[i][0]*(MAX_SINR_DB-MIN_SINR_DB)+MIN_SINR_DB)
                     for i in range(self.num_bs)
                     if loads[i] < self.base_stations[i].max_simultaneous_beams],
                    key=lambda x: x[2], reverse=True)
                if alts:
                    assign_user(alts[0][0], alts[0][1])
                else:
                    user.connected_bs=None; user.beam_id=None
                    user.sinr=MIN_SINR_DB; user.throughput=0
                    outage+=1; user.outage_count+=1

        # ---- Move drones (after UE assignment so reward is meaningful) ----
        prev_ho = self.step_metrics['handover_count'][-1] if self.step_metrics['handover_count'] else 0
        self._move_drones(prev_ho, ho_count)

        # ---- Metrics ----
        self.cmab.total_reward   += total_reward
        self.cmab.optimal_reward += self.num_ues * 0.5
        regret = self.cmab.optimal_reward - self.cmab.total_reward

        avg_sinr  = total_sinr / max(connected,1)
        covered_ues = sum(1 for u in self.users if u.connected_bs is not None and u.sinr > 0.0)
        coverage  = (covered_ues / self.num_ues) * 100.0
        avg_load  = np.mean([bs.load for bs in self.base_stations])

        if tputs:
            cell_edge = np.percentile(tputs, 5)
            self.throughput_distribution.extend(tputs)
            self.sinr_samples.extend(sinrs)
        else:
            cell_edge = 0

        outage_prob   = (outage/self.num_ues)*100
        pingpong_rate = (pingpong/max(ho_count,1))*100 if ho_count>0 else 0

        # Drone handover reduction vs baseline
        if self._baseline_ho_rate is None and self.step >= 20:
            self._baseline_ho_rate = np.mean(self.step_metrics['handover_count'][-20:])
        if self._baseline_ho_rate and self._baseline_ho_rate > 0:
            drone_ho_red = (1 - ho_count/self._baseline_ho_rate)*100
        else:
            drone_ho_red = 0.0

        # Drone positions
        drone_positions = [(bs.x, bs.y, bs.z) for bs in self.base_stations if bs.is_drone]

        # Drone policy rewards
        drone_rewards = []
        for bs in self.base_stations:
            if bs.is_drone and bs.policy and bs.policy.episode_returns:
                drone_rewards.append(bs.policy.episode_returns[-1])

        # â”€â”€ Battery fleet metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        drones = [bs for bs in self.base_stations if bs.is_drone]
        avg_batt    = np.mean([d.battery_pct for d in drones]) if drones else 1.0
        n_active    = sum(1 for d in drones if d.drone_state == DRONE_ACTIVE)
        n_charging  = sum(1 for d in drones if d.drone_state == DRONE_CHARGING)
        n_rtb       = sum(1 for d in drones if d.drone_state == DRONE_RTB)
        step_energy = sum(
            max(d.last_energy_draw_w, 0) * STEP_DURATION_S / 3600.0
            for d in drones
        )
        total_power_w = sum(
            max(d.last_energy_draw_w, 0) for d in drones
        )
        fleet_eff = (total_tput / max(total_power_w, 1.0))  # Mbps/W

        metrics = {
            'avg_sinr': avg_sinr, 'total_throughput': total_tput,
            'handover_count': ho_count, 'coverage': coverage,
            'regret': regret, 'avg_load': avg_load,
            'cell_edge_rate': cell_edge, 'outage_prob': outage_prob,
            'pingpong_rate': pingpong_rate,
            'drone_handover_reduction': drone_ho_red,
            'drone_positions': drone_positions,
            'drone_rewards': np.mean(drone_rewards) if drone_rewards else 0.0,
            # battery
            'avg_battery_pct':        avg_batt,
            'drones_active':          n_active,
            'drones_charging':        n_charging,
            'drones_rtb':             n_rtb,
            'total_energy_consumed_wh': step_energy,
            'fleet_efficiency':       fleet_eff,
        }

        for k in self.step_metrics:
            if k in metrics:
                self.step_metrics[k].append(metrics[k])

        self.step += 1
        if self.step%50==0:
            self._log(f'SINR:{avg_sinr:.1f}dB TP:{total_tput:.0f}M '
                      f'HO:{ho_count} DroneHO_red:{drone_ho_red:.1f}%')

        return metrics

    # ------------------------------------------------------------------ #
    #  RUN                                                                  #
    # ------------------------------------------------------------------ #

    def run_simulation(self, num_steps: int = 200) -> Dict:
        self.reset()
        method_names = {
            'reinforce': 'REINFORCE with Baseline',
            'a2c': 'Advantage Actor-Critic (A2C)',
            'soft_a2c': 'SAC (Soft Actor-Critic)'
        }
        policy_name = method_names.get(self.policy_method.lower(), self.policy_method)
        
        print(f"\n{'='*80}")
        print(f"  6G GROUND HETNET + DRONE BS (Policy Gradient RL)")
        print(f"{'='*80}")
        print(f"  Static: {self.num_macro}M + {self.num_micro}m + {self.num_pico}p")
        print(f"  Drones: {self.num_drones} ({policy_name})")
        print(f"  UEs: {self.num_ues} | Steps: {num_steps}")
        print(f"  Drone max speed: {self.drone_max_speed} m/step")
        print(f"  Drone altitude: {self.drone_z_min}-{self.drone_z_max} m")
        print(f"  PG update interval: every {self.drone_update_interval} steps")
        print(f"{'='*80}\n")

        for s in range(num_steps):
            m = self.simulation_step()
            if (s+1)%50==0:
                cumulative_ho = int(np.sum(self.step_metrics['handover_count']))
                print(f"Step {s+1:3d}/{num_steps} | "
                      f"SINR:{m['avg_sinr']:5.1f}dB | "
                      f"TP:{m['total_throughput']:7.0f}Mbps | "
                      f"HO:{m['handover_count']:3d} (cum {cumulative_ho:4d}) | "
                      f"DroneHOâ†“:{m['drone_handover_reduction']:5.1f}% | "
                      f"Cov:{m['coverage']:5.1f}%")

        drone_pols = [bs for bs in self.base_stations if bs.is_drone]
        total_pg_updates = sum(len(d.policy.episode_returns) for d in drone_pols)

        summary = {
            'total_steps': num_steps,
            'policy_method': self.policy_method,
            'avg_sinr':   np.mean(self.step_metrics['avg_sinr']),
            'avg_throughput': np.mean(self.step_metrics['total_throughput']),
            'total_handovers': int(np.sum(self.step_metrics['handover_count'])),
            'avg_coverage': np.mean(self.step_metrics['coverage']),
            'final_regret': self.step_metrics['regret'][-1],
            'avg_outage_prob': np.mean(self.step_metrics['outage_prob']),
            'avg_pingpong': np.mean(self.step_metrics['pingpong_rate']),
            'avg_cell_edge_rate': np.mean(self.step_metrics['cell_edge_rate']),
            'drone_ho_reduction_final': self.step_metrics['drone_handover_reduction'][-1],
            'total_pg_updates': total_pg_updates
        }
        return summary

    def print_summary(self):
        print(f"\n{'='*80}")
        print("6G + DRONE BS SIMULATION SUMMARY")
        print(f"{'='*80}")
        print(f"\n  Avg SINR:           {np.mean(self.step_metrics['avg_sinr']):.2f} dB")
        print(f"  Avg Throughput:     {np.mean(self.step_metrics['total_throughput']):.0f} Mbps")
        print(f"  Avg Cell-Edge:      {np.mean(self.step_metrics['cell_edge_rate']):.1f} Mbps")
        print(f"  Total Handovers:    {int(np.sum(self.step_metrics['handover_count']))}")
        print(f"  Avg Ping-Pong:      {np.mean(self.step_metrics['pingpong_rate']):.1f}%")
        print(f"  Avg Coverage:       {np.mean(self.step_metrics['coverage']):.1f}%")
        print(f"  Avg Outage:         {np.mean(self.step_metrics['outage_prob']):.2f}%")

        print(f"\n  â”€â”€ DRONE BATTERY & LIFECYCLE â”€â”€")
        print(f"  Avg fleet SoC:       {np.mean(self.step_metrics['avg_battery_pct'])*100:.1f}%")
        print(f"  Avg active drones:   {np.mean(self.step_metrics['drones_active']):.1f} / {self.num_drones}")
        print(f"  Avg charging drones: {np.mean(self.step_metrics['drones_charging']):.1f}")
        print(f"  Fleet efficiency:    {np.mean(self.step_metrics['fleet_efficiency']):.2f} Mbps/W")
        total_energy = sum(self.step_metrics['total_energy_consumed_wh'])
        print(f"  Total energy used:   {total_energy:.2f} Wh (all drones, all steps)")

        print(f"\n  â”€â”€ PER-DRONE REPORT â”€â”€")
        for bs in self.base_stations:
            if not bs.is_drone: continue
            uptime_pct = 100 * bs.total_active_steps / max(
                bs.total_active_steps + bs.total_charging_steps + bs.total_rtb_steps, 1)
            print(f"  Drone {bs.id:2d}: state={bs.drone_state:8s} "
                  f"SoC={bs.battery_pct*100:5.1f}%  "
                  f"cycles={bs.charge_cycles}  "
                  f"uptime={uptime_pct:.0f}%  "
                  f"energy_used={bs.energy_consumed_wh:.1f}Wh  "
                  f"charged={bs.energy_charged_wh:.1f}Wh")
        print(f"{'='*80}\n")

