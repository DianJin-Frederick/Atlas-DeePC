# Robot and Computing Recommendation

**Planning horizon:** Fall 2026  
**Goal:** Deploy data-driven and data-enabled predictive-control algorithms on a physical robot, beginning with contact-rich planar pushing.

## Recommended setup

Purchase a **Franka Research 3**, a dedicated real-time control computer, an NVIDIA GPU workstation, an RGB-D camera, and a safe tabletop workcell. Continue using **MuJoCo** for controller development and add **Isaac Sim** for perception, synthetic data, ROS 2 integration, and digital-twin experiments.

## Robot: Franka Research 3

Franka Research 3 is well suited to control-theory research because it provides:

- Seven degrees of freedom and torque sensing at every joint.
- A 1 kHz external research interface.
- Joint torque, position, and velocity control.
- Joint state, torque, external-force, and collision measurements.
- C++, ROS 2, MATLAB, and Simulink integration.
- Integrated gripper options and hand-guided demonstration collection.

These capabilities support data-driven control, contact-rich manipulation, imitation learning, and safe collection of high-rate input-output trajectories. Before purchase, request an academic quote and a demonstration using the group's own external controller and logger. Confirm sustained 1 kHz operation, timestamps, latency, safety behavior, ROS 2 compatibility, licensing, warranty, and delivery time.

## Simulation strategy

Use both simulators rather than replacing MuJoCo:

| MuJoCo | Isaac Sim |
|---|---|
| Fast controller development | Camera and depth simulation |
| Large multi-seed evaluations | Synthetic visual data |
| Contact and dynamics experiments | Scene and sensor randomization |
| System identification studies | ROS 2 and digital-twin integration |

The current MuJoCo `Pusher-v5` result is a strong starting point, but real pushing will introduce friction uncertainty, compliance, actuator filtering, communication delays, and sensor noise. Add these effects before deployment. Isaac Sim is valuable when the project introduces visual perception or imitation learning, but it does not remove the need for conservative real-robot experiments.

## Computers

### Simulation and learning workstation

Recommended configuration:

- Ubuntu 24.04, subject to Franka software compatibility.
- RTX 5080-class or better GPU.
- At least 16 GB GPU memory; 24--32 GB is preferable for visual learning.
- 64--128 GB system memory.
- Modern Ryzen 9 or Core i9 processor.
- 2--4 TB NVMe storage.

Isaac Sim 6.0 lists an RTX 4080 and 32 GB RAM as its minimum and an RTX 5080 with 64 GB RAM as a good configuration. GPUs without ray-tracing cores, including A100 and H100, are not supported for Isaac Sim.

### Real-time control computer

Use a separate computer for physical experiments:

- Franka-supported Ubuntu release and real-time or low-latency kernel.
- High-clock CPU, 32--64 GB RAM, and 1 TB SSD.
- Dedicated Ethernet adapter and direct connection to the robot controller.
- No simulation or training workloads during robot operation.

A large GPU is unnecessary on this machine. Deterministic timing, reliable networking, logging, and safe shutdown are the priorities.

## Additional equipment

- Franka controller, research interface, gripper, and enabling device.
- External emergency stop, stable table, pedestal, and mounting plate.
- Simple rigid pushing tool with replaceable contact surfaces.
- Overhead RGB-D camera, rigid mount, and calibration target.
- Safe test objects and interchangeable surfaces with different friction.
- Physical guarding or a controlled workcell.
- Spare cables, vendor training, support, and data-backup storage.

## Deployment plan

1. Replace the generic Pusher model with a Franka model in MuJoCo.
2. Add delays, noise, action-rate limits, friction variation, and disturbances.
3. Import the Franka description into Isaac Sim and connect it through ROS 2.
4. Verify the physical robot's communication timing, safety limits, and logging.
5. Begin with low-speed free-space control, then use a rigid tool for planar pushing.
6. Collect safe, persistently exciting real trajectories and compare them with simulation.
7. Deploy the data-driven controller under an independent safety supervisor.
8. Add cameras and imitation learning after the state-based control pipeline is reliable.

## Sources

- [Franka Research 3](https://franka.de/franka-research-3-arm)
- [Franka Research 3 datasheet](https://franka.de/hubfs/Datasheet%20Franka%20Research%203_R02212_2.4_EN.pdf?hsLang=en)
- [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Sim ROS 2 integration](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/ros2_landing_page.html)
- [MuJoCo overview](https://mujoco.readthedocs.io/en/stable/overview.html)

---

*Confirm current specifications, software compatibility, academic pricing, safety requirements, and delivery times with Franka before procurement.*
