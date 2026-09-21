"""Shared configuration for Pusher data collection, fitting, and control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PusherAtlasConfig:
    """Configuration shared by collection, atlas fitting, and control."""

    seed: int = 2026
    offline_episodes: int = 72
    episode_steps: int = 180
    charts_per_phase: int = 8
    t_ini: int = 2
    t_f: int = 8
    svd_energy: float = 0.99999
    min_chart_rank: int = 120
    max_chart_rank: int = 200
    coefficient_scale: float = 3.0
    direction_feature_weight: float = 4.0
    direction_candidate_margin: float = 0.06
    chart_exit_ratio: float = 1.10
    chart_switch_advantage: float = 1.05
    chart_minimum_dwell_steps: int = 0
    rollout_steps: int = 240
    test_seeds: tuple[int, ...] = (101, 102, 103, 104, 105)

    # Offline pose teacher. It is used only to create contact-rich data.
    approach_steps: int = 18
    online_approach_steps: int = 12
    ik_period: int = 3
    ik_restarts: int = 8
    teacher_kp: float = 15.0
    teacher_kd: float = 3.0
    teacher_dither: float = 0.03
    precontact_offset: float = 0.12
    push_contact_offset: float = 0.02
    stratify_reset_pool_multiplier: int = 8
    collection_precontact_offsets: tuple[float, ...] = (0.10, 0.12, 0.14, 0.12, 0.12)
    collection_push_offsets: tuple[float, ...] = (0.015, 0.020, 0.025, 0.020, 0.020)
    collection_dither_scales: tuple[float, ...] = (0.025, 0.040, 0.055, 0.080, 0.045)
    collection_recovery_variants: tuple[bool, ...] = (False, False, False, False, True)
    collection_warmup_steps: tuple[int, ...] = (0, 0, 0, 6, 12)
    collection_warmup_torque_scale: float = 0.85
    recovery_start_step: int = 70
    recovery_steps: int = 4
    recovery_torque_scale: float = 0.75
    # A latched alternate U-tool orientation avoids a wrist singularity when
    # pushing objects from the far-right part of the reset distribution.
    collection_orientation_turn_threshold: float = -0.65
    collection_orientation_turn_radians: float = 1.5707963267948966
    coverage_probe_episodes: int = 12
    coverage_probe_seed: int = 9001

    # Online phase manager and geometric reference.
    approach_tolerance: float = 0.075
    contact_lost_distance: float = 0.19
    push_speed: float = 0.16
    success_radius: float = 0.08
    success_hold_steps: int = 8

    # QP weights.
    q_posture_weight: float = 25.0
    q_velocity_weight: float = 0.06
    tip_weight_approach: float = 350.0
    tip_weight_push: float = 180.0
    object_weight_approach: float = 8.0
    object_weight_push: float = 6000.0
    object_velocity_weight: float = 4.0
    input_weight: float = 0.025
    input_rate_weight: float = 0.15
    terminal_multiplier: float = 5.0
    torque_limit: float = 2.0
    lam_ini: float = 5.0e4
    lam_g: float = 2.0e-3
    lam_behavior_anchor: float = 500.0
    behavior_reference_weight: float = 0.0
    policy_direction_weight: float = 8.0
    policy_prior_weight: float = 1000.0
    policy_chart_selection_weight: float = 2.0
    policy_hidden_width: int = 192
    policy_max_iterations: int = 180
    policy_ensemble_size: int = 1
    # Chosen on the disjoint 131--160 validation block; this member avoided the
    # branch averaging observed in the three-model joint-target ensemble.
    policy_random_seed: int = 2003

    @classmethod
    def smoke(cls) -> "PusherAtlasConfig":
        """Small configuration suitable for executing the notebook quickly."""
        return cls(
            offline_episodes=8,
            episode_steps=140,
            charts_per_phase=3,
            t_f=6,
            min_chart_rank=100,
            max_chart_rank=128,
            rollout_steps=220,
            test_seeds=(101,),
            ik_restarts=3,
            policy_ensemble_size=1,
        )
