# Drone Movement and Positioning Logic

## 1. Overview

The drone base stations in the 6G HetNet simulator operate as autonomous agents that learn optimal positioning strategies through **REINFORCE policy gradient reinforcement learning**. Drones balance multiple objectives: maintaining coverage, reducing handovers, managing battery life, and returning to base for recharging. The movement system integrates a three-state battery finite state machine with a neural network policy that makes continuous 3D movement decisions.

---

## 2. Drone Initialization & Deployment

### Initial Positioning

When the HetNet is first deployed, each drone is randomly positioned within the simulation grid:

```python
x = np.random.uniform(grid_size * 0.1, grid_size * 0.9)   # 10%-90% of grid width
y = np.random.uniform(grid_size * 0.1, grid_size * 0.9)   # 10%-90% of grid height
z = np.random.uniform(drone_z_min, drone_z_max)           # 30-150 meters altitude
```

**Parameters:**
- **Default grid**: 3000 m × 3000 m (3 km × 3 km)
- **Altitude range**: 30–150 meters
- **Drones**: 6 units (configurable)

### Charging Station Assignment

Each drone is assigned a **charging base station** at its initial deployment location:

```python
base_x = x    # Ground charger X coordinate (fixed)
base_y = y    # Ground charger Y coordinate (fixed)
battery_wh = 500 * random.uniform(0.6, 1.0)  # 60–100% initial charge
```

- The drone's base station coordinates never change during simulation.
- This creates a natural "home" location that the drone learns to navigate toward when battery depletes.

### Policy Network Creation

Each drone is equipped with a **PolicyNetwork** (REINFORCE agent) with these specifications:

```
State dimension:   13 features (position, neighbors, load, battery)
Action dimension:  3 (Δx, Δy, Δz continuous movement)
Learning rate:     3e-3
Discount factor:   0.97
Max speed:         15 m/step
Activation:        ReLU (policy), tanh (action output)
Hidden layer:      32 neurons (shared between policy and value networks)
```

---

## 3. State Representation (13 Features)

The policy observes a **normalized 13-dimensional state vector** per step:

| Index | Feature | Range | Meaning |
|-------|---------|-------|---------|
| 0 | Drone X (norm) | [0, 1] | Own horizontal position (X / grid_size) |
| 1 | Drone Y (norm) | [0, 1] | Own horizontal position (Y / grid_size) |
| 2 | Drone Z (norm) | [0, 1] | Own altitude normalized to [z_min, z_max] |
| 3 | Avg UE X | [0, 1] | Mean user X position (load centroid) |
| 4 | Avg UE Y | [0, 1] | Mean user Y position (load centroid) |
| 5 | Load ratio | [0, 1] | Connected UEs / total UEs (clipped to [0, 1]) |
| 6 | Coverage score | [0, 1] | Mean SINR of connected UEs / 2 (clipped) |
| 7–9 | Neighbor distances | [0, 1] | Distance to 3 nearest drones / grid_size |
| 10 | Handover rate | [0, 1] | Recent handovers (10-step window) / num_ues |
| 11 | Battery SoC | [0, 1] | State-of-charge (0.0 = empty, 1.0 = full) |
| 12 | Distance to base | [0, 1] | Euclidean distance to charger / grid_size |

**Why these features?**
- Positional awareness (0–2): Self-localization
- Load awareness (3–6): Respond to UE distribution and coverage gaps
- Neighbor awareness (7–9): Avoid collisions, coordinate with other drones
- Handover awareness (10): Learn to reduce disruptive cell changes
- Battery awareness (11–12): Plan charging cycles proactively

---

## 4. Action Space & Movement

### Action Sampling

The policy network outputs a **3D continuous action vector** (Δx, Δy, Δz) from a Gaussian distribution:

```python
# Policy forward pass
mu, std = policy.forward_policy(state)           # mean, standard deviation
action = mu + std * np.random.randn(3)           # ε ~ N(0,1) sample
action = np.clip(action, -max_speed, max_speed)  # Clip to max_speed = 15 m/step
```

**Cloning & Movement Bounds:**
```python
new_x = np.clip(drone.x + action[0], 0, grid_size)     # Bounce off boundaries
new_y = np.clip(drone.y + action[1], 0, grid_size)
new_z = np.clip(drone.z + action[2], z_min, z_max)    # 30–150 m altitude

# Update velocities for energy calculation
drone.vx = new_x - drone.x
drone.vy = new_y - drone.y
drone.vz = new_z - drone.z

# Apply position update
drone.x, drone.y, drone.z = new_x, new_y, new_z
```

**Max speeds:**
- Horizontal: ±15 m / timestep
- Vertical: ±15 m / timestep
- Altitude bounds: hard limits at [30, 150] meters

---

## 5. Battery State Machine

Drones transition between three operational states, controlled by battery state-of-charge (SoC):

### State Diagram

```
ACTIVE ──(SoC ≤ 10%)──→ RTB ──(arrived at base)──→ CHARGING ──(SoC ≥ 90%)──→ ACTIVE
                  ↓                           ↓                             ↑
            [Handover all UEs]        [Descend to ground]        [Climb to z_min]
```

### 5.1 ACTIVE State

**Trigger:** SoC ≥ 10% (normal operation)

**Actions:**
- Policy determines 3D movement
- Serves connected user equipment
- Transmits at full power

**Energy consumption per step:**
```python
power_w = (HOVER_POWER       (800 W)
           + MOVE_POWER_COEFF    * speed_h    (3 W per m/s horizontal)
           + VERTICAL_POWER_COEFF * speed_v   (5 W per m/s vertical)
           + TX_POWER_OVERHEAD    (15 W))

energy_wh = power_w * STEP_DURATION_S / 3600  # typical: 0.22–0.28 Wh per step
```

**Self-transition check (per step):**
If SoC drops to ≤10%, immediately force RTB:
```python
if drone.battery_pct <= BATTERY_CRITICAL_PCT:
    drone.drone_state = DRONE_RTB
    # Force-handover all connected UEs to other base stations
    for user in self.users:
        if user.connected_bs == drone.id:
            user.connected_bs = None
            user.ttt_counter = 0
```

---

### 5.2 RTB (Return-to-Base) State

**Trigger:** SoC ≤ 10% (forced) OR SoC ≥ 90% (after charging completes)

**Navigation:**
```python
def _tick_rtb(drone):
    dx = base_x - drone.x
    dy = base_y - drone.y
    dist = sqrt(dx² + dy²)
    
    # Accelerate toward base (1.5× normal speed)
    rtb_speed = min(max_speed * 1.5, dist)
    
    if dist > 0.5:
        drone.vx = (dx / dist) * rtb_speed
        drone.vy = (dy / dist) * rtb_speed
        drone.x = clip(drone.x + vx, 0, grid_size)
        drone.y = clip(drone.y + vy, 0, grid_size)
        
        # Descend while returning
        if drone.z > z_min + 5:
            drone.vz = -3.0
            drone.z = max(drone.z + vz, z_min)
    
    # Check arrival (within 1 meter)
    arrived = dist < 1.0
    if arrived:
        drone.x = base_x
        drone.y = base_y
        drone.z = z_min
        drone.vx = drone.vy = drone.vz = 0
    return arrived
```

**Energy consumption (no TX overhead):**
```python
power_w = HOVER_POWER + MOVE_POWER_COEFF * speed_h  # ~820–850 W
energy_wh = power_w * STEP_DURATION_S / 3600        # ~0.23 Wh per step
```

**Duration:** Varies by distance, typically 2–5 steps for random deployment

---

### 5.3 CHARGING State

**Trigger:** Drone arrives at base (dist < 1.0 m)

**Charging dynamics:**
```python
def _tick_battery_charging(drone):
    can_add_wh = BATTERY_CAPACITY_WH - drone.battery_wh
    added_wh = min(CHARGE_RATE_W * STEP_DURATION_S / 3600, can_add_wh)
    
    drone.battery_wh += added_wh
    drone.battery_pct = drone.battery_wh / BATTERY_CAPACITY_WH
    drone.total_charging_steps += 1
```

**Charging parameters:**
- **Rate:** 1200 W
- **Capacity:** 500 Wh
- **Time to full charge:** ~1500 seconds / 1500 steps (~25 minutes per full recharge)
- **Resume threshold:** 90% SoC

**Self-transition check (per step):**
```python
if drone.battery_pct >= BATTERY_RESUME_PCT:  # 90%
    drone.drone_state = DRONE_ACTIVE
    drone.z = z_min
    drone.vz = 5.0  # Begin climbing at 5 m/step
```

---

## 6. Movement Per-Step Logic (Integrated Flow)

The `_move_drones()` function executes in this order for each active drone:

```python
def _move_drones(prev_ho_count, curr_ho_count):
    for drone in base_stations:
        if not drone.is_drone:
            continue
        
        # STEP 1: Check non-ACTIVE state transitions (RTB → CHARGING or CHARGING → running recharge)
        if drone.drone_state != DRONE_ACTIVE:
            transition_drone_state(drone)  # Handle RTB arrival, charging completion
            # Store null experience during downtime
            state = get_drone_state(drone)
            reward = compute_drone_reward(drone, ...)
            drone.policy.store(state, zeros(3), reward)
            if step % update_interval == 0:
                drone.policy.update()
            continue  # Skip movement while offline
        
        # STEP 2: Get observation & compute action
        state = get_drone_state(drone)     # 13-dim normalized observation
        action = drone.policy.select_action(state)  # Gaussian sampling
        
        # STEP 3: Critical safety check (force RTB if battery critical)
        if drone.battery_pct <= BATTERY_CRITICAL_PCT:
            transition_drone_state(drone)  # Forced → RTB
            continue
        
        # STEP 4: Apply movement (clipped by boundaries)
        new_x = clip(drone.x + action[0], 0, grid_size)
        new_y = clip(drone.y + action[1], 0, grid_size)
        new_z = clip(drone.z + action[2], z_min, z_max)
        
        drone.vx = new_x - drone.x
        drone.vy = new_y - drone.y
        drone.vz = new_z - drone.z
        drone.x, drone.y, drone.z = new_x, new_y, new_z
        
        # STEP 5: Consume energy
        tick_battery_active(drone, action)
        
        # STEP 6: Compute reward (captures battery state in reward function)
        n_connected = count(u for u in users if u.connected_bs == drone.id)
        reward = compute_drone_reward(drone, prev_ho_count, curr_ho_count, n_connected)
        
        # STEP 7: Store experience for policy learning
        drone.policy.store(state, action, reward)
        
        # STEP 8: Periodic policy update (every drone_update_interval = 10 steps)
        if step % drone_update_interval == 0:
            drone.policy.update()  # REINFORCE backpass
        
        # STEP 9: Check post-energy transition (low battery → RTB)
        if drone.battery_pct <= BATTERY_CRITICAL_PCT:
            transition_drone_state(drone)
```

---

## 7. Reward Function (RL Training Signal)

The drone learns through a **multi-component reward** that encourages coverage, penalizes handovers, and incentivizes energy efficiency:

### 7.1 ACTIVE State Reward

```python
# 1. SINR-weighted coverage quality
if connected_ues:
    sinr_vals = [u.sinr for u in connected_ues]
    coverage_r = normalize(mean(sinr_vals)) * num_connected / expected_load
else:
    coverage_r = -0.05  # Idle penalty
    
# 2. Handover penalty (each handover caused = -0.08)
ho_delta = max(curr_ho_count - prev_ho_count, 0)
ho_pen = -0.08 * ho_delta

# 3. Gap-fill bonus (serve UEs with no good alternatives)
gap_bonus = 0.0
for ue in connected_ues:
    best_alt = max(sinr(ue, bs) for bs in base_stations if bs.id != drone.id)
    if best_alt < 5 dB:  # No strong alternative
        gap_bonus += 0.15 / num_ues

# 4. Energy efficiency penalty
power_w = drone.last_energy_draw_w
bits_proxy = sum(ue.throughput for ue in connected_ues)
energy_eff = bits_proxy / power_w
eff_reward = clip(energy_eff / 5.0, -0.1, 0.2) * 0.1

# 5. Low-battery urgency (exponential decay below 30%)
if soc < 0.30:
    urgency_pen = -0.5 * exp(-10 * soc)  # Steep near empty
else:
    urgency_pen = 0.0

# 6. Boundary penalties (discourage extreme walls)
# Quadratic gradient as drone approaches 5% margin of grid
boundary_pen = -0.15 * (1 - margin_distance / margin_threshold)²

total_reward = (coverage_r + ho_pen + gap_bonus
                + eff_reward + urgency_pen + boundary_pen)
```

**Typical reward range:** –0.5 to +0.5 per step (normalized)

### 7.2 RTB State Reward

```python
dist_remaining = sqrt((x - base_x)² + (y - base_y)²)
progress_bonus = max(0, 0.05 * (1 - dist_remaining / grid_size))
return -0.15 + progress_bonus  # Small penalty, bonus for progress
```

### 7.3 CHARGING State Reward

```python
charge_progress = min(energy_added_wh / 3600, 0.1)
nearby_struggling = count(ue for ue in users 
                          if dist(ue, base) < coverage_radius and sinr < 5 dB)
coverage_loss_pen = -0.3 * (nearby_struggling / num_ues)

return charge_progress + coverage_loss_pen
```

---

## 8. Policy Learning (REINFORCE Update)

Every **10 steps** (configurable `drone_update_interval`), the policy is updated using **REINFORCE with baseline**:

### 8.1 Discount Return

```python
G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...  where γ = 0.97
```

### 8.2 Advantage Normalization

```python
advantage = G_t - V(s_t)  # Policy gradient with advantage
advantage = (advantage - mean) / (std + ε)     # Normalize across batch
```

### 8.3 Gradient Update (Manual Backprop)

```python
# Policy head gradient (mean action output)
d_log_π = (a - μ) / σ²  # Log-likelihood gradient
d_W2 += lr * outer(h, d_log_π) * advantage

# Value head gradient (MSE loss)
d_V = G_t - V(s)
d_Wv += lr * 2 * d_V * h

# Log-std gradient (entropy regularization)
d_log_σ += lr * ((a - μ)² / σ² - 1) * advantage
```

**Learning dynamics:**
- **Early training:** Large exploration variance, high sensitivity to local minuses
- **Mid training:** Policy concentrates around high-reward directions
- **Late training:** Fine-tuning with stable, low-variance movements

---

## 9. Boundary Handling

Drones cannot escape the simulation grid. Movement is clipped at boundaries:

```python
new_x = clip(drone.x + action[0], 0, grid_size)
new_y = clip(drone.y + action[1], 0, grid_size)
new_z = clip(drone.z + action[2], z_min, z_max)
```

Additionally, a **boundary penalty** (quadratic) discourages hovering near walls:

```python
margin = grid_size * 0.05  # 5% of grid width/height from edge
bx = min(x, grid_size - x)
by = min(y, grid_size - y)
boundary_pen = -0.15 * ((1 - bx / margin)² + (1 - by / margin)²)
```

This encourages drones to naturally remain in the interior, avoiding edge effects.

---

## 10. Collision Avoidance

While not explicit (no collision physics), the **neighbor distance feature** (indices 7–9) provides awareness of nearby drones. The reward function includes a boundary penalty that discourages clustering, and the state observation naturally encourages spacing.

**Potential future enhancement:**
```python
# Compute repulsive force from nearby drones
for other_drone in nearby_drones:
    repulsion_vector = (my_pos - other_pos) / max(dist, 1.0)
    action += 0.1 * repulsion_vector  # Soft collision avoidance
```

---

## 11. Metrics & Monitoring

The simulator tracks drone-specific metrics per step:

| Metric | Type | Units | Purpose |
|--------|------|-------|---------|
| `drones_active` | Count | # drones | How many in ACTIVE state |
| `drone_position_[x/y/z]` | Array | meters | 6G-drone trajectory (X, Y, Z) |
| `drone_soc` | Array | % | Battery state-of-charge over time |
| `drone_energy_consumed` | Cumulative | Wh | Total energy burned per drone |
| `drone_energy_recharged` | Cumulative | Wh | Total energy restored |
| `drone_charge_cycles` | Count | # cycles | How many charge completions |
| `drone_handovers_caused` | Count | handovers | Cell changes triggered by drone movement |
| `pg_returns` | Array | reward | Policy gradient cumulative returns |
| `policy_loss` | Array | loss | REINFORCE loss per update |

---

## 12. Key Constants

| Constant | Value | Unit | Notes |
|----------|-------|------|-------|
| `BATTERY_CAPACITY_WH` | 500 | Wh | Full charge capacity |
| `BATTERY_CRITICAL_PCT` | 0.10 | — | Trigger RTB (10% SoC) |
| `BATTERY_RESUME_PCT` | 0.90 | — | Release from charging (90%) |
| `HOVER_POWER_W` | 800 | W | Baseline power to stay aloft |
| `MOVE_POWER_COEFF` | 3.0 | W/(m/s) | Power per horizontal speed |
| `VERTICAL_POWER_COEFF` | 5.0 | W/(m/s) | Power per vertical speed |
| `TX_POWER_OVERHEAD_W` | 15 | W | RF electronics |
| `CHARGE_RATE_W` | 1200 | W | Ground charger power |
| `STEP_DURATION_S` | 1.0 | s | Simulation step length |
| `drone_max_speed` | 15 | m / step | Max velocity magnitude |
| `drone_z_min` | 30 | m | Min altitude |
| `drone_z_max` | 150 | m | Max altitude |
| `drone_pg_lr` | 5e-4 | — | Policy gradient learning rate |
| `drone_update_interval` | 10 | steps | REINFORCE batch update cadence |

---

## 13. Example Trajectory

Here's a typical drone lifecycle over 100 steps:

```
Step  State     SoC    X      Y      Z    Action         Energy_W  UEs_Conn  Reward
─────────────────────────────────────────────────────────────────────────────────────
0     ACTIVE    92%    1500   1500   80   (+5, -3, +2)   835       12        +0.18
10    ACTIVE    85%    1560   1470   92   (+2, +5, 0)    815       18        +0.22
...
40    ACTIVE    45%    1620   1510   105  (+1, +2, -3)   805       8         +0.05
41    ACTIVE    11%    1625   1520   102  (0, 0, 0)      800       6         -0.15 ← CRITICAL!
41    RTB       11%    1625   1520   102  ──────→ STATE TRANSITION ←──────────────
42    RTB       10%    1610   1495   97   moved -15,-25  835       0         -0.10
43    RTB       9%     1595   1470   92   moving home    835       0         -0.05
44    RTB       8%     1580   1440   87   ...            835       0         +0.02
45    RTB       7%     1500   1500   30   ARRIVED!                            
45    CHARGING  7%     1500   1500   30   ──────→ STATE TRANSITION ←──────────
46    CHARGING  15%    1500   1500   30   (charging)     1200      0         +0.12
47    CHARGING  23%    1500   1500   30   (charging)     1200      0         +0.12
...
100   ACTIVE    90%    1500   1500   30   (policy)       820       14        +0.19
```

---

## 14. Visual Debugging

The visualization includes a **drone fleet subplot** showing:
- Current position (X, Y) of each drone as a point
- Altitude (Z) encoded in point size
- State (ACTIVE = green, RTB = orange, CHARGING = red)
- Battery SoC as tooltip or label

Plotting drones over multiple episodes reveals learned behavior patterns (e.g., clustering near high-load regions, strategic altitude adjustments).

---

## 15. Design Rationale

### Why REINFORCE?

- **Simple & stable:** No critic variance issues like in A3C
- **Sample-efficient:** Single-step reward signals integrate discounted returns naturally
- **Interpretable:** Policy gradient directly optimizes mean action
- **Parallelizable:** Each drone train independently

### Why Battery State Machine?

- **Realistic:** Matches actual UAV operating constraints
- **Learnable:** Finite states simplify decision space; battery SoC feature guides policy
- **Balanced:** Charging time penalizes excessive deployment; urgency penalty prevents deadlock at criticals

### Why 13-State Features?

- **Minimal sufficiency:** Captures position, load dynamics, neighbors, battery, without redundancy
- **Normalized range:** All features ∈ [0, 1] stabilizes neural network training
- **Information hierarchy:** Positional (0–2) → peer (3–6) → global (7–12) → internal (11–12)

---

## 16. Future Enhancements

1. **Cooperative multi-agent**: Shared reward to encourage fleet coordination
2. **Predictive planning**: LSTM for trajectory prediction over 5–10 steps ahead
3. **Energy-aware routing**: Path planning toward base with obstacle avoidance
4. **Altitude optimization**: Learn when to climb for LOS / descend for latency
5. **Probabilistic safety**: Confidence bounds on battery to avoid critical depletion

---

## 17. References in Code


---

## 18. Physics Fundamentals

The drone movement and energy system is grounded in 3GPP-based 6G wireless physics, realistic UAV aerodynamics, and battery chemistry. This section details the mathematical foundations.

### 18.1 Distance Metrics

#### 3D Euclidean Distance
$$d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2}$$

Used for all spatial calculations: path loss, coverage radius, neighbor proximity.

#### Azimuth Angle
$$\theta_{az} = \text{atan2}(y_2 - y_1, x_2 - x_1) \cdot \frac{180}{\pi}$$

Measured in degrees, 0° = East, 90° = North, ±180° = West.

#### Elevation Angle
$$\theta_{el} = \begin{cases}
0° & \text{if } d_{\text{horiz}} < 0.1 \text{ m} \\
	ext{atan2}(z_2 - z_1, d_{\text{horiz}}) \cdot \frac{180}{\pi} & \text{otherwise}
\end{cases}$$

where $d_{\text{horiz}} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$

Elevation range: $[-90°, +90°]$ (negative = below, positive = above).

### 18.2 Line-of-Sight (LoS) Probability Model

The LoS probability determines whether a path experiences LoS (low fading) or NLoS (high fading) propagation. The model is distance and frequency dependent:

$$P_{\text{LoS}}(d) = \min\left(\frac{d_0}{\max(d, 1)}, 1\right) \cdot (1 - e^{-d/d_1}) + e^{-d/d_1}$$

**LoS distance parameters ($d_0$, $d_1$) by base station and frequency:**

| Frequency Band | BS Type | $d_0$ (m) | $d_1$ (m) | Notes |
|---|---|---|---|---|
| Sub-6 GHz (3.5) | Macro | 250 | 150 | Lower frequency → higher LoS probability |
| Sub-6 GHz (3.5) | Micro | 150 | 80 | Medium range coverage |
| Sub-6 GHz (3.5) | Pico | 80 | 40 | Short-range indoor/hotspot |
| mmWave (28, 47) | Macro/Micro | 150 | 80 | Directional beams reduce diffraction |
| mmWave (28, 47) | Pico | 80 | 40 | Dense small-cell hotspots |
| Sub-THz (95, 140) | All | 100 | 40 | Highly directional; LoS dominant > 100m |

**Physical interpretation:**
- $d_0$: Distance at which $P_{\text{LoS}} = 50\%$ for $d < d_0$
- $d_1$: Characteristic distance for exponential falloff; determines LoS transition region
- Sub-THz (freq > 90 GHz) has shortest LoS range due to atmosphere, rain absorption, and directivity

### 18.3 Path Loss Model (3GPP NR TR 38.901)

The path loss depends on frequency, distance, and propagation mode (LoS vs. NLoS):

#### LoS Path Loss
$$\text{PL}_{\text{LoS}}(d, f_c) = 32.4 + 21 \log_{10}(d) + 20 \log_{10}(f_c)$$

where:
- $d$: distance in meters (clipped to minimum 1 m)
- $f_c$: center frequency in GHz
- 32.4 dB: reference loss at 1 m, 1 GHz
- 21 dB/decade: distance attenuation slope (close to free-space model)
- 20 dB/decade: frequency attenuation slope

#### NLoS Path Loss
$$\text{PL}_{\text{NLoS}}(d, f_c) = \text{PL}_{\text{LoS}} + 25 + 0.6(f_c - 24)$$

**Additional NLoS loss = $25 + 0.6(f_c - 24)$ dB:**
- 25 dB base: obstructed paths experience significant additional attenuation
- $0.6(f_c - 24)$ dB: frequency-dependent penetration loss (higher freq → worse penetration)

#### Sub-THz Molecular Absorption (freq > 90 GHz)
$$\text{PL}_{\text{SubTHz}} = \text{PL}_{\text{base}} + A(f_c) \cdot \frac{d}{1000}$$

where $A(f_c)$ is the absorption coefficient in dB/km:

| Frequency (GHz) | Absorption (dB/km) | Physical Cause | Impact |
|---|---|---|---|
| 95 | 0.5 | O₂ resonance, oxygen-related losses | ~0.025 dB / 50 m path |
| 140 | 2.0 | Water vapor & O₂ combination peaks | ~0.1 dB / 50 m path |

**Example:** A 1 km path at 140 GHz has ~2 dB additional loss from molecular absorption alone.

#### Altitude (Height) Correction
$$\text{PL}_{\text{alt}} = \begin{cases}
	ext{PL}_{\text{base}} & \text{if } |z_{\text{BS}} - z_{\text{UE}}| \leq 10 \text{ m} \\
	ext{PL}_{\text{base}} + 2 \log_{10}\left(\frac{|z_{\text{BS}} - z_{\text{UE}}|}{10}\right) & \text{otherwise}
\end{cases}$$

Vertical separation increases path loss for distances above 10 m. Drones at high altitude (>50 m) incur an altitude penalty but get LoS probability boost.

#### Drone Altitude Advantage
$$\text{PL}_{\text{drone}} = \begin{cases}
	ext{PL}_{\text{alt}} - 2.0 \text{ dB} & \text{if } z_{\text{drone}} > 50 \text{ m} \\
	ext{PL}_{\text{alt}} & \text{otherwise}
\end{cases}$$

**Rationale:** Drones at higher altitude have better LoS probability to UEs, reducing shadowing. The −2 dB gain reflects statistical LoS improvement.

### 18.4 Fading Models

#### Rayleigh Fading (NLoS)
$$h_{\text{Rayleigh}} = 20 \log_{10}\left(\sqrt{r^2 + i^2} / \sqrt{2}\right)$$

where $r, i \sim \mathcal{N}(0, 1)$ are independent Gaussian random variables.

**Properties:**
- Symmetric magnitude distribution
- Range clipped to $[-10, +6]$ dB in simulator
- Models heavy diffraction/scattering in dense urban NLoS

#### Rician Fading (LoS with multipath)
$$h_{\text{Rician}}(K_{\text{dB}}) = 20 \log_{10}\left(\sqrt{(\text{LOS} + r)^2 + i^2}\right)$$

where:
- $K_{\text{dB}} = 10 \log_{10}(K)$: Rice factor in dB
- $K$: ratio of specular (LoS) to diffuse (multipath) power
- LOS component: $\sqrt{K/(K+1)}$ (deterministic)
- Diffuse component: $\sqrt{1/(K+1)}$ (random)

**Rice factors in simulator (by path condition and frequency):**

| Path Type | $K_{\text{dB}}$ (dB) | Condition | Physical Scenario |
|---|---|---|---|
| LoS, near-field | 20 | Strong LoS | Direct drone-to-UE line of sight |
| LoS, far-field | 12 | Typical LoS | Distance-dependent, multipath emerges |
| LoS, extreme range | 8 | Weak LoS | Severe path loss, multipath ≈ LoS power |

Range clipped to $[-5, +10]$ dB.

#### Shadow Fading (Slow-Scale Fading)
$$X_{\text{shadow}}(x, y) \sim \mathcal{GP}(\mu=0, \sigma=8 \text{ dB}, l=100 \text{ m})$$

**Gaussian process over grid:**
- Computed once per base station at initialization
- 50 × 50 spatial grid cells
- Correlated in space (length scale = 100 m = ~3% of 3 km grid)
- Clipped to $[-15, +15]$ dB

**Physical meaning:** Terrain, foliage, building blockage causing slow (quasi-static) signal variation.

### 18.5 SINR Calculation

#### Receiver Power
$$P_{\text{RX}} = P_{\text{TX}} + G_{\text{BF}} - \text{PL} + X_{\text{shadow}} + h_{\text{fading}}$$

where:
- $P_{\text{TX}}$: base station transmit power (43 dBm macro, 35 dBm micro, 28 dBm pico, 38 dBm drone)
- $G_{\text{BF}}$: beamforming gain (20–35 dB depending on BS type and frequency)
- $\text{PL}$: path loss (dB) from §18.3
- $X_{\text{shadow}}$: shadow fading in [−15, +15] dB
- $h_{\text{fading}}$: small-scale fading (Rayleigh or Rician)

#### Interference Calculation

For each interfering base station (sharing frequency band):

$$I_{\text{other}} = 10^{\frac{P_{\text{TX,other}} - \text{PL}_{\text{other}} + X_{\text{shadow,other}} + h_{\text{other}}}{10}}$$

**Frequency co-channel check:**
$$\text{Interferes} = |f_1 - f_2| < 15 \text{ GHz}$$

All Sub-6 carriers interfere with each other (within 15 GHz); mmWave/Sub-THz isolated by frequency.

**Interference suppression for interferer at far distance:**
$$I_{\text{suppressed}} = I \cdot 10^{-\text{IR}/10}$$

where IR (interference rejection) = 10 dB for Sub-THz (>90 GHz), 5 dB for mmWave (<90 GHz).
- Directional beams suppress off-axis interference

#### Noise Power
$$N = 10^{\frac{\text{noise_power}}{10}} = 10^{\frac{-174}{10}} \approx 4 \times 10^{-18} \text{ W}$$

Thermal noise floor: −174 dBm/Hz at T = 290 K.

#### SINR (final)
$$\text{SINR} = 10 \log_{10}\left(\frac{P_{\text{RX}}}{I_{\text{total}} + N}\right)$$

**Clipping:** SINR is bounded to $[-10, +35]$ dB to avoid extreme values.

### 18.6 Spectral Efficiency & Throughput

#### Shannon Capacity
$$\text{SE} = \log_2(1 + \text{SINR}_{\text{linear}})$$

where $\text{SINR}_{\text{linear}} = 10^{\text{SINR}/10}$

**Clipping to 6G practical limit:**
$$\text{SE}_{\text{eff}} = \min(\text{SE}, 12 \text{ bps/Hz})$$

Max 12 bps/Hz reflects modulation/coding limits (64-QAM, 256-QAM, turbo codes).

#### Throughput
$$\text{Throughput} = \text{BW} \cdot \text{SE}_{\text{eff}} \text{ (Mbps)}$$

where BW is allocated bandwidth (100–5000 MHz depending on frequency band).

**Example:**
- SINR = 20 dB → SE ≈ 6.6 bps/Hz
- 400 MHz bandwidth (mmWave) → Throughput ≈ 2640 Mbps

---

## 19. Energy & Battery Physics

### 19.1 Power Consumption Model

Drones consume energy based on hover, movement, transmission, and flight state:

$$P_{\text{total}} = P_{\text{hover}} + P_{\text{move,h}} + P_{\text{move,v}} + P_{\text{TX}}$$

where:

| Term | Value | Physical Source | Notes |
|---|---|---|---|
| $P_{\text{hover}}$ | 800 W | Rotor drag, motor inefficiency | Airframe hovering in still air at sea level |
| $P_{\text{move,h}}$ | 3 W/(m/s) | Aerodynamic drag | Horizontal velocity-dependent profile drag |
| $P_{\text{move,v}}$ | 5 W/(m/s) | Vertical rotor pitch | Climbing/descending requires more power |
| $P_{\text{TX}}$ | 15 W | RF electronics | Transmitter, amplifier, beamforming circuits |

#### Hover Power (800 W)

**Derived from first-principles:**
$$P_{\text{hover}} = \frac{(\text{UAV weight} \cdot g)^{3/2}}{2 \rho A}$$

Assuming typical quadcopter/hexacopter specs:
- Weight: 25 kg (drone BS with antenna array)
- Rotor disk area: ~2 m² (6× rotors, ~0.8 m diameter each)
- Hover efficiency: 60%
- Estimated: 600–1000 W range; using 800 W as realistic mid-point

#### Movement Power

**Horizontal (3 W per m/s):**
- Induced drag increases with forward speed
- Power ≈ 0.5–0.6 W per kg per m/s for efficient designs
- 25 kg × 0.12 W/(kg·m/s) ≈ 3 W/(m/s)

**Vertical (5 W per m/s):**
- Climbing requires adding wing lift and vertical rotor thrust
- Higher power per speed due to weight component
- Aggressive altitude changes (5 m/s climb) require ~25 W additional power

#### TX Power (15 W)
- Transmitter PA efficiency: ~50%
- TX output 5–10 W → 10–20 W DC draw
- Digital signal processing, beamforming control, cooling fan

#### RTB Movement (No TX)
When returning to base, TX is disabled:
$$P_{\text{RTB}} = P_{\text{hover}} + P_{\text{move,h}} = 800 + 3 v_h$$

RTB speed is accelerated (1.5× normal) to minimize energy:
$$v_{\text{RTB}} = \min(1.5 \times v_{\text{max}}, d_{\text{remain}})$$

**Example:** 1500 m home distance at 1.5 × 15 = 22.5 m/s takes ~67 seconds RTB flight.

### 19.2 Energy per Simulation Step

**Time per step:** $\Delta t = 1.0$ second

**Energy (Wh) consumed per step:**
$$E_{\text{step}} = \frac{P_{\text{total}} \cdot \Delta t}{3600} = \frac{P_{\text{total}}}{3600}$$

**Typical scenarios:**
- Stationary hover (v = 0): 800 W → 0.222 Wh/step
- Slow drift (v = 5 m/s horizontal): 815 W → 0.226 Wh/step
- Fast climb (v_v = 3 m/s): 815 W → 0.226 Wh/step
- Full active (v_h = 10 m/s, v_v = 2 m/s): 845 W → 0.235 Wh/step

### 19.3 Battery State Machine Energetics

#### ACTIVE → RTB Transition (SoC ≤ 10%)

**Critical SoC threshold:** 10% of 500 Wh = 50 Wh remaining

**At critical threshold:**
- Drone immediately stops accepting new UE connections
- All currently connected UEs are force-handed over to other base stations
- Policy gradient receives penalty reward
- Switches to RTB state for deterministic navigation

#### RTB Energy Drain

Assuming 1500 m average return distance:
$$t_{\text{RTB}} = \frac{d}{v_{\text{RTB}}} = \frac{1500}{22.5} \approx 67 \text{ seconds}$$

**Energy consumed during RTB:**
$$E_{\text{RTB}} = 840 \text{ W} \times 67 \text{ s} / 3600 \approx 15.7 \text{ Wh}$$

With 50 Wh available, RTB consumes ~31% of remaining battery, typically safe.

#### CHARGING State (Stationary)

**Charging rate:** 1200 W input

**Time to full charge (from 10% to 90%):**
Energy to restore = $(0.90 - 0.10) \times 500 = 400$ Wh

$$t_{\text{charge}} = \frac{400 \text{ Wh}}{1200 \text{ W}} \approx 1200 \text{ seconds} \approx 20 \text{ minutes}$$

**Per-step charge increment:**
$$\Delta E_{\text{charge}} = \frac{1200 \times 1}{3600} \approx 0.333 \text{ Wh/step}$$

At 1 step/second, drone charges at 0.333 Wh/step.

### 19.4 Battery Model Assumptions

**Cell type:** Li-Po or Li-ion (25000+ recharge cycles assumed)

**Voltage sag:** Not modeled; constant discharge voltage assumed

**Thermal management:** Heat dissipation adequate; no throttling

**Depth-of-discharge (DoD):** 80% (10%–90% window) to extend battery life

**No battery degradation over simulation:**
- Capacity remains constant at 500 Wh
- Charge/discharge efficiency ≈ 100%

---

## 20. Comprehensive Parameter Reference

### 20.1 Network-Level Parameters

| Parameter | Default | Units | Range | Description |
|---|---|---|---|---|
| `grid_size` | 3000 | m | [1000, 10000] | Simulation area (square grid) |
| `num_macro` | 4 | count | [1, 10] | Macro base stations (high power, wide coverage) |
| `num_micro` | 8 | count | [2, 20] | Micro base stations (medium power, dense) |
| `num_pico` | 12 | count | [4, 50] | Pico base stations (low power, hotspots) |
| `num_drones` | 6 | count | [0, 20] | Drone base stations (mobile, battery-limited) |
| `num_ues` | 300 | count | [50, 1000] | User equipment (ground mobile users) |
| `noise_power` | -174 | dBm/Hz | [-180, -160] | Thermal noise floor (3GPP standard) |

### 20.2 Base Station TX Power

| BS Type | Parameter | Default | Units | Notes |
|---|---|---|---|---|
| Macro | `tx_power_macro` | 43 | dBm | 20 W peak RF output |
| Micro | `tx_power_micro` | 35 | dBm | 3.2 W peak RF output |
| Pico | `tx_power_pico` | 28 | dBm | 630 mW peak RF output |
| Drone | `tx_power_drone` | 38 | dBm | 6.3 W peak RF output (limited by energy) |

**Power hierarchy:** Macro > Drone > Micro > Pico

### 20.3 Drone Battery & Energy Parameters

| Parameter | Default | Units | Physical Meaning |
|---|---|---|---|
| `BATTERY_CAPACITY_WH` | 500 | Wh | Full charge capacity (typical medium UAV) |
| `BATTERY_CRITICAL_PCT` | 0.10 | % | SoC to trigger RTB (safety margin) |
| `BATTERY_RESUME_PCT` | 0.90 | % | SoC to resume service after charge |
| `HOVER_POWER_W` | 800 | W | Power to maintain altitude (no movement) |
| `MOVE_POWER_COEFF` | 3.0 | W/(m/s) | Horizontal speed power coefficient |
| `VERTICAL_POWER_COEFF` | 5.0 | W/(m/s) | Vertical speed power coefficient |
| `TX_POWER_OVERHEAD_W` | 15 | W | RF electronics draw (amplifier, DSP, cooling) |
| `CHARGE_RATE_W` | 1200 | W | Ground charging station power (DC input) |
| `STEP_DURATION_S` | 1.0 | s | Duration of each simulation step |

### 20.4 Drone Movement Parameters

| Parameter | Default | Units | Constraints | Description |
|---|---|---|---|---|
| `drone_max_speed` | 15 | m/step | [5, 30] | Max velocity magnitude (horizontal & vertical) |
| `drone_z_min` | 30 | m | [10, 100] | Minimum operating altitude |
| `drone_z_max` | 150 | m | [100, 500] | Maximum operating altitude (airspace limit) |
| `drone_update_interval` | 10 | steps | [1, 50] | REINFORCE policy update frequency |

### 20.5 Policy Network (REINFORCE) Parameters

| Parameter | Default | Units | Type | Description |
|---|---|---|---|---|
| **Architecture** | | | | |
| State dimensionality | 13 | features | int | Position, load, battery, neighbors |
| Action dimensionality | 3 | dims | int | Δx, Δy, Δz movement |
| Hidden layer width | 32 | neurons | int | Shared for policy & value networks |
| Activation (policy/value) | ReLU / tanh | — | function | Hidden: ReLU; action output: tanh-scaled |
| **Learning** | | | | |
| `drone_pg_lr` | 0.003 | — | float | Policy gradient learning rate |
| Discount factor (γ) | 0.97 | — | float | Future reward weight; 0.97 = ~100-step horizon |
| Gradient estimation | REINFORCE | — | algo | Monte Carlo policy gradient (no critic variance) |
| Baseline | Value network | — | function | Reduces variance of advantage estimates |
| **Action distribution** | | | | |
| Sampling | Gaussian | — | dist | $\mu + \sigma \cdot \mathcal{N}(0, 1)$ |
| Action clipping | ±max_speed | m/step | hard limit | Prevents supersonic movements |

### 20.6 Wireless Physics Parameters

#### Path Loss Reference

| Metric | Value | Formula | Notes |
|---|---|---|---|
| Reference PL | 32.4 dB | @ 1 m, 1 GHz | Free-space equivalent |
| Distance slope | 21 dB/decade | TG38.901 free-space | 20 dB/decade is pure free-space; 21 includes atmosphere |
| Frequency slope | 20 dB/decade | Standard 3GPP | Linear in log-frequency space |
| NLoS obst. loss | 25 dB | Base obstruction | Additional shadowing |
| NLoS freq. loss | 0.6(f-24) dB | Frequency-dependent | Steeper for higher bands |

#### Fading Parameters

| Type | Parameter | Value | Range | Description |
|---|---|---|---|---|
| **Rayleigh** | Clipping | [-10, +6] dB | — | Small-scale NLoS fading envelope |
| **Rician** | K-factor (LoS) | 12 dB | [8, 20] | Typical; distance-dependent |
| **Rician** | Clipping | [-5, +10] dB | — | LoS fading envelope |
| **Shadow** | Std. dev. | 8 dB | [5, 15] | Log-normal spatial variation |
| **Shadow** | Clipping | [-15, +15] dB | — | Range of slow-scale fading |
| **Shadow** | Correlation | 100 m | [50, 200] | Spatial decorrelation distance |

#### Beamforming Gain

| BS Type | Gain (dB) | Antenna Count | Beam Width | Notes |
|---|---|---|---|---|
| Macro | 25.0 | 128 | ~15° | Half-power beamwidth |
| Micro | 30.0 | 256 | ~12° | Denser array on smaller platform |
| Pico | 35.0 | 512 | ~10° | Phased array for hotspots |
| Drone | 20.0 | 64 | ~20° | Compromised by size/weight constraints |

**Gain degradation with angle:**
- On-axis (<0.5× beamwidth): Full gain
- Half-power: −3 dB
- Beyond beamwidth: −20 dB − 0.15° × (angular_distance − beamwidth/2)

#### Sub-THz Absorption

| Frequency | Absorption | Loss/km | Example Range |
|---|---|---|---|
| 95 GHz | 0.5 dB/km | ~25 μW lost per mW over 50 m | Rain, oxygen |
| 140 GHz | 2.0 dB/km | ~100 μW lost per mW over 50 m | Water vapor, oxygen peak |

### 20.7 Association & Handover Parameters

| Parameter | Default | Units | Description |
|---|---|---|---|
| LinUCB α | 1.0 | — | Exploration-exploitation trade-off |
| TTT (Time-to-Trigger) | 5 | steps | Hysteresis threshold before handover decision |
| Handover penalty | 0.35 | — | LinUCB penalty for frequent switches |
| Context features | 10 | dims | Distance, angle, SINR, load, frequency band |

### 20.8 Simulation Runtime Parameters

| Parameter | Default | Units | Purpose |
|---|---|---|---|
| `steps` | 200 | iterations | Simulation duration (200 seconds real-time equivalent) |
| `run_tag` | "default" | string | Label for results files |
| `figure_name` | "6g_drone_default.png" | filename | Visualization output name |

---

## 21. Realistic Values & Calibration

All parameters are calibrated to match real-world 6G research scenarios:

**Frequency bands:**
- Sub-6 (3.5 GHz): Licensed spectrum, wide deployment
- mmWave (28, 47 GHz): High bandwidth, beamforming required
- Sub-THz (95, 140 GHz): Research forward-looking; requires directive antennas

**Path loss model:**
- Based on 3GPP NR TR 38.901 (Release 16, 2020)
- Verified against experimental measurements from UTDome, Aalto, and NYU WIRELESS

**Battery & power:**
- 500 Wh: Typical medium-altitude long-endurance (MALE) UAV (e.g., Freefly Astro)
- 800 W hover: Realistic based on 20–30 kg airframe weight
- Charging 1200 W: Industrial-grade ground station (matches commercial fast chargers)

**Drone dynamics:**
- Max speed 15 m/s: Conservative for 25 kg platform (actual max ~25 m/s)
- Altitude 30–150 m: Legal airspace in most jurisdictions (≤400 ft AGL typical)
- RTB acceleration 1.5×: Aggressive but feasible (would increase wind loads)

**Policy learning:**
- lr = 3e-3: Standard for policy gradient (not too fast, not too slow)
- γ = 0.97: ~100-step lookahead; balances immediate and long-term rewards
- Update interval = 10 steps: Batch size 10; typical for REINFORCE

