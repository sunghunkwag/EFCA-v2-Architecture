# EFCA-v2 Redesign: Detailed Structure and Integration Specification

**Objective:** Redesign a detailed architecture to implement measurable functional self-awareness (self-uncertainty modeling, internal parameter adjustment, self-directed goal generation) integrated within a single agent and enables end-to-end learning.

**Core Philosophy:** Based on predictive processing theory with hierarchical free energy minimization, securing learning stability and adaptability through metacognitive control, and inducing self-directed behavior through intrinsic motivation.

---

## 1. System Overview and Information Flow

EFCA-v2 has the following core modules that interact organically:

1. **Environment:** External world or simulation. Provides state `s_t`, receives action `a_t`, and returns next state `s_{t+1}` and reward `r_t^{ext}`.
2. **Sensor Preprocessing:** Receives raw sensor input `x_raw` (e.g., images, sounds) and converts it to the first layer input format `x_0` for H-JEPA (resizing, normalization, etc.).
3. **Hierarchical JEPA (H-JEPA):** Multi-layer structure that receives sensor input to generate abstract representations `z_k` (k=1...K), and calculates prediction errors `ΔE_k` by predicting the representations of the next state at each level. A core element of free energy minimization.
4. **Tokenizer & Sparse Global Workspace (s-GWT):** Tokenizes high-level representations (`z_K`) from H-JEPA and other salient information (e.g., large prediction errors), routing them to sparsely selected slots to create the workspace state `S_gwt` that corresponds to the current integrated "contents of consciousness."
5. **Continuous-Time Liquid Neural Network (CT-LNN):** Receives hierarchical representations `{z_k}` from H-JEPA, GWT state `S_gwt`, previous action `a_{t-1}` to model the continuous dynamic state `h(t)` of the system. The basis for short-term prediction and action decisions.
6. **Task Policy Network (Task Policy `π_task`):** Determines action `a_t` to interact with the current environment based on the CT-LNN state `h(t)` and GWT state `S_gwt`. Trained to maximize external reward `r_t^{ext}`.
7. **Bipartite Metacognition:**
   * **Probe Network:** Monitors internal system states (LNN state `h(t)`, JEPA errors `ΔE_k`, current meta parameters, etc.) to output metacognitive assessments `φ` (uncertainty `φ_unc`, competence `φ_comp`, effort `φ_eff`).
   * **Meta-Controller (Meta-Controller `π_meta`):** Receives the probe's output `φ` to determine meta-actions `a_meta` that adjust internal system parameters (JEPA layer weights `λ_k`, learning rate `α`, exploration rate `ε_explore`, GWT sparsity, etc.). Trained to maximize intrinsic motivation rewards.
8. **Dual-Axis Intrinsic Motivation:** Defines reward `r_meta` for meta-controller learning. Based on two axes: knowledge seeking (uncertainty reduction) and competence improvement (task performance improvement or progress toward internal goal achievement).
9. **Memory Mesh:** Compresses important experiences (high error, high reward, meta-controller intervention points, etc.) using VQ-VAE for long-term storage, and retrieval/use when needed (e.g., replay buffer, self-modeling).

**Information Flow Diagram (Conceptual):**

```mermaid
graph TD
    subgraph Environment
        EnvState[State s_t]
        EnvAction[Action a_t]
        EnvNextState[Next State s_{t+1}]
        EnvReward[External Reward r_t^{ext}]
    end

    subgraph Agent
        Sensor[Sensor Preprocessing]
        HJEPA[Hierarchical JEPA (K layers)]
        Tokenizer[Tokenizer]
        GWT[Sparse Global Workspace (s-GWT)]
        CTLNN[Continuous-Time LNN]
        TaskPolicy[Task Policy π_task]
        Probe[Probe Network]
        MetaControl[Meta-Controller π_meta]
        Motivation[Dual-Axis Intrinsic Motivation]
        Memory[Memory Mesh (VQ-VAE)]

        EnvState -- x_raw --> Sensor -- x_0 --> HJEPA
        HJEPA -- {z_k}, {ΔE_k} --> CTLNN
        HJEPA -- z_K, ΔE_k? --> Tokenizer -- tokens --> GWT
        GWT -- S_gwt --> CTLNN
        GWT -- S_gwt --> TaskPolicy
        CTLNN -- h(t) --> TaskPolicy
        CTLNN -- h(t) --> Probe
        TaskPolicy -- a_t --> EnvAction
        EnvAction -- a_{t-1} --> CTLNN

        HJEPA -- {ΔE_k} --> Probe
        MetaControl -- Current Params --> Probe
        Probe -- φ (unc, comp, eff) --> MetaControl
        MetaControl -- a_meta (λ_k, ε_explore, α...) --> HJEPA & TaskPolicy & Optimizer & GWT & CTLNN
        Motivation -- r_meta --> MetaControl

        Motivation -- Uses φ_unc, φ_comp, r_t^{ext}? --> Motivation

        Memory -- Store Significant Events --> Memory
        Memory -- Retrieve for Replay/Context --> HJEPA & CTLNN & TaskPolicy?

        EnvReward --> TaskPolicy  // For Task RL update
        EnvReward -- Maybe used by --> Motivation
        HJEPA -- Prediction Errors --> Motivation // For Epistemic Reward
    end

    EnvAction --> EnvNextState
    EnvAction --> EnvReward
```

(Note: Mermaid diagrams are visually rendered in markdown viewers/editors that support them.)

## 2. Mathematical Redefinition and Detailed Development

### 2.1 Hierarchical Free Energy (H-JEPA Focus)

Each JEPA layer k (from 1 to K) tries to minimize the following free energy term F_k:

$$
F_k(t) = E_{q(z_k|c_k)}[
\underbrace{D_{pred}(z_k, \hat{z}_k)}_\text{Energy: prediction accuracy}
+ \beta_k
\underbrace{KL[q(z_k|c_k) \parallel p(z_k)]}_\text{Entropy: regularization/prior information}
]
$$

(1a)

$z_k$: Latent representation vector of layer k (Encoder output). Encoder_k(x_{k-1}). $x_0$ is preprocessed sensor input.

$\hat{z}_k$: Predicted next-time latent representation in layer k (Predictor output). Predictor_k(z_{k-1}, c_k'). Context c_k' may include lower layer information or time information.

$Target(z_k)$: Actual latent representation calculated at the next time point $t+\Delta t_k$ (using a non-learning target encoder). The prediction target.

$D_{pred}(z_k, \hat{z}_k)$: Prediction loss function. E.g., $|\text{sg}(Target(z_k)) - \hat{z}_k|_2^2$. sg is stop-gradient.

$q(z_k|c_k)$: Inference distribution of latent representation z_k given the current context c_k (e.g., z_{k-1}) (encoder models this).

$p(z_k)$: Prior distribution of latent representation z_k (e.g., standard normal distribution N(0, I)).

$\beta_k$: Hyperparameter that adjusts the weight between the energy term and entropy term.

The overall free energy objective is the sum of the free energies of each layer, weighted by weights ($\lambda_k(t)$) adjusted by the meta-controller:

$$
F_{JEPA}(t) = \sum_{k=1}^{K} \lambda_k(t) F_k(t)
$$

(1b)

$\lambda_k(t) \in [10^{-3}, 1]$: Layer-specific weights dynamically adjusted by the meta-controller (π_meta). Initial value is 1/K.

### 2.2 Continuous-Time Model (CT-LNN) and Learning

CT-LNN dynamics:

$$
\dot{h}(t) = f_\theta(h(t), u(t)) \text{ where } u(t) = \text{concat}(\{z_k(t)\}_{k=1..K}, S_{gwt}(t), a_{t-1})
$$

(2a)

$h(t)$: Hidden state vector of CT-LNN.

$f_\theta$: Neural network with parameters $\theta$ (e.g., Liquid Time-Constant Networks, Gated Recurrent Unit ODE).

$u(t)$: Control input at the current time point. Includes hierarchical representations, workspace state, and previous action.

Learning: CT-LNN primarily contributes to representation learning for task performance. It can therefore be learned by backpropagating the loss (L_task) of the task policy network π_task, or it can have its own short-term prediction loss (L_lnn_pred). Backpropagation using the Adjoint Sensitivity Method (similar to original Eq. 2, but with a different objective loss):

$$
\nabla_\theta L_{task} = \int_{t_0}^{t_1} \bar{\lambda}(t') \left( \frac{\partial L_{task}}{\partial h(t')} \frac{\partial h(t')}{\partial \theta} + \frac{\partial L_{task}}{\partial u(t')} \frac{\partial u(t')}{\partial h(t')} \frac{\partial h(t')}{\partial \theta} \right) dt'
$$

(2b)

$L_{task}$: Task-related loss (e.g., RL value loss or policy loss).

$\bar{\lambda}(t')$: Time discount factor (e.g., $e^{(t_1-t')/\tau_b}$).

### 2.3 Meta-Controller Policy and Reward (Bipartite Metacognition & Dual-Axis Motivation)

Probe Network:

$$
\phi_t = \text{ProbeNet}(\text{sg}(h(t)), \{\text{sg}(\Delta E_k(t))\}_{k=1..K}, \{\lambda_k(t)\}, \alpha(t), \epsilon_{explore}(t), ...)
$$

(3a)

$\phi_t = (\phi_{unc}, \phi_{comp}, \phi_{eff})$: Metacognitive state vector (uncertainty, competence, effort estimates).

$\phi_{unc} \approx \sum_k w_k' \Delta E_k(t)$ (weighted sum of prediction errors)

$\phi_{comp} \approx V_{task}(h(t), S_{gwt}(t))$ (task value function estimate)

$\phi_{eff} \approx \text{computational metrics}$ (e.g., GWT slot activity, LNN computational load)

sg: Stop-gradient. The probe only observes and does not affect backpropagation.

Meta-Controller Policy:

$$
a_t^{meta} \sim \pi_{meta}(\cdot|\phi_t) \text{ where } a_t^{meta} = (\Delta\lambda_k, \Delta\alpha, \Delta\epsilon_{explore}, ...)
$$

(3b)

Meta-action a_meta specifies the changes or new values for internal parameters.

Meta-Controller Learning (Actor-Critic):

$$
L_{meta} = E_t[(r_t^{meta} + \gamma_{meta} V_{meta}(\phi_{t+1}) - V_{meta}(\phi_t))^2 - \eta A_t^{meta} \log \pi_{meta}(a_t^{meta}|\phi_t)]
$$

(3c)

$V_{meta}$: Meta value function.

$A_t^{meta}$: Meta advantage function ($r_t^{meta} + \gamma_{meta} V_{meta}(\phi_{t+1}) - V_{meta}(\phi_t)$).

$\eta$: Entropy regularization weight.

Dual-Axis Intrinsic Reward (r_meta):

$$
r_t^{meta} = w_{epistemic} \cdot r_t^{epistemic} + w_{competence} \cdot r_t^{competence}
$$

(3d)

Knowledge Axis (Epistemic): Uncertainty reduction reward.

$r_t^{epistemic} = -\phi_{unc}(t)$ (current uncertainty itself is cost/negative reward)

or $r_t^{epistemic} = \phi_{unc}(t-1) - \phi_{unc}(t)$ (amount of uncertainty reduction)

or $r_t^{epistemic} = \text{KL divergence reduction in JEPA}$ (amount of information gain)

Competence Axis (Competence): Goal achievement or performance improvement reward.

$r_t^{competence} = \phi_{comp}(t) - \phi_{comp}(t-1)$ (amount of estimated competence increase)

or $r_t^{competence} = r_t^{ext}$ (directly use external task reward)

or $r_t^{competence} = \text{Progress towards internal sub-goal}$

$w_{epistemic}, w_{competence}$: Weights determining the relative importance of the two axes (fixed or learnable).

## 3. Architecture Component Detailed Specifications

| Module                 | Input                                                            | Core Structure                                                                | Output                                                              | Key Parameters/Features                                                            |
| ---------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Sensor Preprocessing   | Raw x_raw (B, C, H, W)                                           | Resizing, normalization (ImageNet statistics, etc.)                           | x_0 (B, C', H', W')                                                 | Target H', W'                                                                      |
| H-JEPA Layer k         | z_{k-1} (or x_0 for k=1)                                         | Encoder (ConvNext-T), Predictor (Cross-Attention), Target Encoder (EMA)       | z_k (B, d_k), prediction $\hat{z}_k$, prediction error ΔE_k         | Latent dim d_k, $\beta_k$, EMA decay τ_ema                                         |
| Tokenizer              | z_K (B, d_K), High ΔE_k?                                         | VQ-VAE Encoder or K-Means                                                     | Token Sequence (B, T_tok, d_tok)                                    | Codebook size, d_tok                                                               |
| s-GWT                  | Token Sequence (B, T_tok, d_tok)                                 | Router (selects K_slot slots), Slot Memory (N_slot x d_slot), Cross-Attention | S_gwt (B, d_gwt = K_slot * d_slot)                                  | N_slot (total), K_slot (active), d_slot, Attention heads                           |
| CT-LNN                 | {z_k}, S_gwt, a_{t-1}                                            | LTC/GRU-ODE Cells (e.g., 4 cells, torchdiffeq), MLP projection                | h(t) (B, d_h), Optional: $\hat{z}_k(t+\Delta t)$                    | Hidden dim d_h, ODE solver (e.g., rk4), integration step Δt_i                      |
| Task Policy π_task     | h(t), S_gwt                                                      | MLP or LSTM + Actor-Critic Heads (Value, Policy)                              | Action a_t (Discrete: Softmax / Continuous: Gaussian), Value V_task | Actor/Critic network sizes, Action space definition                                |
| Probe Network          | h(t), {ΔE_k}, λ_k, α, ε_explore                                  | MLP (e.g., 2 layers: 256 -> 128)                                              | φ = (φ_unc, φ_comp, φ_eff) (B, 3)                                   | Layer sizes                                                                        |
| Meta-Controller π_meta | φ                                                                | LSTM (e.g., 64 units) + Actor-Critic Heads (Meta-Value, Meta-Policy)          | a_meta (changes to λ_k, α, ε_explore...), Meta-Value V_meta         | LSTM size, Meta-action space definition (ranges, discretization)                   |
| Memory Mesh            | Significant events (x_t, a_t, r_t^{ext}, h(t), S_gwt, φ_t, ΔE_k) | VQ-VAE Encoder/Decoder                                                        | Compressed trace z_m, Reconstructed event                           | Codebook size, Compression level, Storage mechanism (e.g., FIFO buffer, Vector DB) |

Hardware requirement estimates: (similar to original)

P0-P2 (basic functionality validation): Expected to accommodate 4 x A100 (80GB).

P3-P4 (complex environments and long-term learning): May require 8 x A100 or more (especially when using memory mesh and large-scale GWT).

## 4. Algorithm: Learning Loop

### 4.1 Fast Inner Loop (Millisecond Scale: Action Decision and Short-term Prediction)

```python
# Pseudocode: Inner Loop (High Frequency)
def inner_loop(agent_state, env_state):
    # 1. Perception and State Abstraction (H-JEPA)
    x_0 = preprocess(env_state.raw_observation)
    z = {}
    predictions = {}
    errors = {}
    current_input = x_0
    for k in range(1, K + 1):
        z[k], predictions[k], errors[k] = agent_state.h_jepa[k].forward(current_input, agent_state.meta_params.lambda_k[k])
        current_input = z[k] # Input for next layer

    # 2. Workspace Update (Tokenizer & s-GWT)
    # Select salient info (e.g., z[K], high errors)
    salient_info = select_salient(z[K], errors)
    tokens = agent_state.tokenizer.encode(salient_info)
    S_gwt = agent_state.gwt.forward(tokens)

    # 3. Continuous State Update (CT-LNN)
    # Prepare LNN input: u_t = concat(z.values(), S_gwt, agent_state.last_action)
    u_t = prepare_lnn_input(z, S_gwt, agent_state.last_action)
    h_t = agent_state.ct_lnn.forward(agent_state.h_prev, u_t, dt=config.dt_inner_loop) # ODE integration

    # 4. Action Selection (Task Policy)
    # Prepare policy input: policy_input = concat(h_t, S_gwt)
    policy_input = prepare_policy_input(h_t, S_gwt)
    a_t, V_task = agent_state.task_policy.select_action(policy_input, explore_eps=agent_state.meta_params.epsilon_explore)

    # 5. Environment Step
    next_env_state, reward_ext, done, info = env.step(a_t)

    # 6. Store Transition (for Outer Loop and Memory Mesh)
    transition = Transition(env_state, a_t, reward_ext, next_env_state, done, h_t, S_gwt, z, errors, agent_state.meta_params)
    agent_state.replay_buffer.add(transition)
    if is_significant(transition):
        agent_state.memory_mesh.store(transition)

    # 7. Update Agent State for next step
    agent_state.h_prev = h_t
    agent_state.last_action = a_t

    return agent_state, next_env_state, done
```

### 4.2 Slow Outer Loop (Second Scale: Learning and Meta Control)

```python
# Pseudocode: Outer Loop (Low Frequency / Asynchronously)
def outer_loop(agent_state, optimizer_jepa, optimizer_lnn, optimizer_task, optimizer_meta):
    if len(agent_state.replay_buffer) < config.batch_size:
        return agent_state # Wait for more data

    # 1. Sample Batch from Replay Buffer
    batch = agent_state.replay_buffer.sample(config.batch_size)

    # 2. Calculate Losses

    # 2a. H-JEPA Free Energy Loss (Eq. 1b)
    # Requires forward pass through H-JEPA for the batch (potentially recomputing parts or using stored values)
    total_fe_loss = 0
    lambda_k = agent_state.meta_params.lambda_k # Use current lambda values
    for k in range(1, K + 1):
        # Calculate F_k (Eq. 1a) using stored/recomputed z_k, hat_z_k, targets from batch
        fe_loss_k = calculate_F_k(batch, k, agent_state.h_jepa[k].beta_k)
        total_fe_loss += lambda_k[k] * fe_loss_k
    loss_jepa = total_fe_loss

    # 2b. CT-LNN Loss (Optional: self-prediction, or implicitly via Task Loss)
    # loss_lnn = calculate_lnn_prediction_loss(batch, agent_state.ct_lnn) # If applicable

    # 2c. Task Policy Loss (RL Loss, e.g., Actor-Critic)
    # Requires rollouts or stored values V_task, rewards etc.
    loss_task = calculate_task_rl_loss(batch, agent_state.task_policy, agent_state.ct_lnn) # CT-LNN gradients flow through here

    # 2d. Meta-Controller Loss (RL Loss, Eq. 3c)
    # Requires running ProbeNet on batch states and calculating meta-rewards
    phi_batch = agent_state.probe_net(batch.h, batch.errors, batch.meta_params) # Get meta-states
    meta_rewards = calculate_dual_axis_reward(phi_batch, batch.reward_ext, config.w_epistemic, config.w_competence) # Eq. 3d
    loss_meta = calculate_meta_rl_loss(batch, agent_state.meta_controller, phi_batch, meta_rewards, agent_state.meta_value_net)

    # 3. Compute Gradients and Update Networks

    # Zero Gradients
    optimizer_jepa.zero_grad()
    optimizer_lnn.zero_grad()
    optimizer_task.zero_grad()
    optimizer_meta.zero_grad()

    # Backward Pass (handle shared components like CT-LNN carefully)
    # Option 1: Separate backward passes
    loss_jepa.backward(retain_graph=True) # Retain graph if LNN uses JEPA outputs for its loss/task loss
    # loss_lnn.backward(retain_graph=True) # If LNN has separate loss
    loss_task.backward(retain_graph=True) # Gradients flow back to Task Policy and CT-LNN
    loss_meta.backward() # Gradients flow back to Meta-Controller, Meta-Value, Probe (if not sg)

    # Option 2: Combined loss (might need careful gradient scaling)
    # total_loss = loss_jepa + loss_task + loss_meta # (+ loss_lnn)
    # total_loss.backward()

    # Optimizer Step
    optimizer_jepa.step()
    optimizer_lnn.step() # Updates CT-LNN based on task loss gradients
    optimizer_task.step() # Updates Task Policy
    optimizer_meta.step() # Updates Meta-Controller and Probe

    # Update Target Networks (e.g., for JEPA Target Encoders, RL Target Networks)
    agent_state.update_target_networks()

    # 4. Update Meta-Parameters based on Meta-Controller Action (could happen in inner loop too)
    # This step is tricky - meta-actions from π_meta should be applied.
    # One way: π_meta outputs *target* values for λ_k, ε_explore, α. Update agent_state.meta_params slowly towards these targets.
    # meta_action_target = agent_state.meta_controller.get_target_params(average_phi_over_batch)
    # agent_state.update_meta_params(meta_action_target)

    return agent_state
```

CUDA kernel: retro_grad.cu is still important for efficiently calculating the Adjoint Method (Eq. 2b) of CT-LNN. The built-in Adjoint of torchdiffeq is a good starting point.

## 5. Experimental Setup (Redefined and Added)

### 5.1 P0: Basic Prediction Capability (GridWorld-Lite)

Environment: 16x16 tiles, 4 channels (position, goal, obstacles, noise). Actions: {N,E,S,W,Stay}. Full-state noise p=0.05.

Objective: Achieve short-term prediction PSNR > 28 dB in H-JEPA Layer 0 (most concrete level) (1x A100, within 2 hours). Self-awareness relevance: Validate the ability to learn the basic structure of the environment.

### 5.2 P1: Workspace Throughput and Latency (Latency Stress)

Environment: Synthetic token stream generator. Input vector sequences mimicking H-JEPA z_K to Tokenizer and s-GWT at various speeds (1kHz ~ 20kHz).

Objective: Maintain 90%ile latency from GWT slot update to S_gwt generation below a specific threshold (e.g., 20ms). Self-awareness relevance: Validate the ability to process current state in GWT without information bottlenecks.

### 5.3 P2: Meta-Control Stability (Noisy Dynamics)

Environment: OpenAI Gym "CartPole-v1" or "Pendulum-v1" with periodically changing system parameters (e.g., pole length, friction) or added sensor noise.

Objectives:

- Verify that the meta-controller adaptively adjusts JEPA layer weights (λ_k) according to prediction errors (ΔE_k) (correlation analysis).
- Verify faster and more stable convergence in unstable environments compared to baseline agents using fixed λ_k and learning parameters.

Self-awareness relevance: Validate the ability to recognize uncertainty in internal models (JEPA) through Probe and adjust learning parameters (λ_k) through Meta-Controller.

### 5.4 P3: Intrinsic Motivation Exploration (Sparse Reward / Exploration Challenge)

Environment: Sparse reward environments that are difficult to explore (e.g., "MountainCar-v0", complex maze finding).

Objectives:

- Verify that EFCA-v2 agents using Dual-Axis intrinsic motivation (especially the Epistemic axis) explore environments and solve tasks more efficiently than standard RL agents using only external rewards.
- Analyze if the meta-controller tends to increase ε_explore when φ_unc is high during the exploration phase.

Self-awareness relevance: Validate self-directed behavior that recognizes lack of knowledge (uncertainty) and actively adjusts exploration strategies (ε_explore) to resolve it.

### 5.5 P4: Complex Tasks and Self-Modeling (Complex Task & Long-Term Adaptation)

Environment: Complex simulation environments with multiple objectives and non-stationarity (e.g., simulated robot arm manipulation, resource management games).

Objectives:

- Demonstrate stable performance and adaptability to environmental changes during long-term learning (millions of steps).
- Quantify changes in self-model (Identity Drift) by comparing past experiences stored in Memory Mesh with current states (JSD metric).
- Analyze patterns of the meta-controller adjusting various internal parameters (learning rates, GWT settings, etc.) for long-term performance improvement.

Self-awareness relevance: Evaluate the ability to monitor long-term self-state, modify internal strategies in response to environmental changes, and maintain a stable self-model.

## 6. Checkpoints and Key Metrics

| CP  | Experiment | Metric             | Formula / Target Value (Example)                                                  | Self-Awareness Aspect                     |
| --- | ---------- | ------------------ | --------------------------------------------------------------------------------- | ----------------------------------------- |
| 1   | P0         | JEPA L0 PSNR       | $20\log_{10}(MAX\_I/MSE(\hat{x}_0, x_0)) > 28$                                    | Basic environment modeling capability     |
| 2   | P1         | GWT Latency        | 90%ile processing time < 20 ms @ 10 kHz input                                     | Information processing efficiency         |
| 3   | P2         | λ-Error Corr.      | Pearson Corr($\lambda_k(t)$, $\Delta E_k(t)$) > 0.5 (in expected direction)       | Parameter adjustment based on uncertainty |
| 4   | P2         | Convergence Speed  | Time to reach target performance < 0.7 * Baseline time                            | Meta-control stability contribution       |
| 5   | P3         | Exploration Eff.   | Sparse reward acquisition success rate > Baseline + 30%                           | Exploration based on intrinsic motivation |
| 6   | P3         | Unc.-Explore Corr. | Pearson Corr($\phi_{unc}(t)$, $\epsilon_{explore}(t)$) > 0.4 (when needed)        | Behavior adjustment based on uncertainty  |
| 7   | P4         | Adaptation Perf.   | Non-stationary environment average reward > Baseline + 20%                        | Long-term adaptation and self-correction  |
| 8   | P4         | Identity Drift     | JSD(Hist(SelfHash_pre), Hist(SelfHash_post)) < 0.1 (after 1M steps) @ Memory Mesh | Self-model stability                      |
| 9   | ALL        | Probe Corr (Unc.)  | Pearson Corr($\phi_{unc}$, Actual $\sum \Delta E_k$) > 0.6                        | Metacognitive uncertainty accuracy        |
| 10  | ALL        | Probe Corr (Comp.) | Pearson Corr($\phi_{comp}$, Actual Task Performance) > 0.7                        | Metacognitive competence accuracy         |

## 7. Discussion and Considerations

**Integration and Stability:** The biggest challenge is the stable integration of multiple feedback loops (JEPA, Task RL, Meta RL). The time scales (ms vs sec) and learning rates of each loop must be carefully controlled. Reward design and constraints are important for the meta-controller to contribute to stabilization without amplifying instability.

**Computational Complexity:** H-JEPA, CT-LNN (especially the Adjoint Method), and large-scale GWT require significant computational resources. Efficient implementation (CUDA kernel optimization, model parallelization) is essential.

**Hyperparameter Tuning:** With many modules and complex interactions, the hyperparameter space is very wide. Initially, a strategy of testing each module individually and gradually integrating them is effective. Systematic tuning and validation through P0-P4 experimental design is necessary.

**Memory Mesh Utilization:** Currently designed primarily for replay and analysis, but retrieved past experiences can be used as context inputs to GWT or CT-LNN to enhance long-term dependency modeling or continuous learning capabilities (additional research).

**Definition and Measurement of Self-Awareness:** Need to clarify the relationship between the capabilities that constitute "functional self-awareness" (uncertainty modeling, parameter adjustment, goal generation) and the proposed metrics, and explore more direct self-awareness evaluation methodologies (e.g., asking the agent questions about its own state).

## 8. Conclusion

The redesigned EFCA-v2 provides a concrete and integrated blueprint for functional self-awareness. By combining hierarchical free energy, continuous-time dynamics, global workspace, bipartite metacognition, and dual-axis intrinsic motivation within a single framework, it aims to enable the agent to understand its internal state and actively adjust learning and behavioral strategies. The detailed module specifications