"""Offline intent priors used to regularize the Pusher DeePC QP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .environment import PHASE_NAMES, behavior_output


@dataclass
class OfflineBehaviorPolicy:
    """Smooth action prior trained only from the offline demonstration library."""

    input_scaler: StandardScaler
    action_scaler: StandardScaler
    model: object
    direction_weight: float
    output_kind: str = "action"
    teacher_kp: float = 15.0
    teacher_kd: float = 3.0
    torque_limit: float = 2.0
    uses_orientation_context: bool = False

    def feature(self, state, goal_xy, direction, phase, orientation_turn=0.0):
        phase_one_hot = np.eye(len(PHASE_NAMES))[int(phase)]
        feature = np.r_[
            behavior_output(state, goal_xy),
            self.direction_weight * np.asarray(direction, dtype=float),
            phase_one_hot,
        ]
        if getattr(self, "uses_orientation_context", False):
            feature = np.r_[feature, float(orientation_turn)]
        return feature

    def predict(self, state, goal_xy, direction, phase, orientation_turn=0.0):
        feature = self.feature(
            state, goal_xy, direction, phase, orientation_turn
        )[None, :]
        models = self.model if isinstance(self.model, (list, tuple)) else [self.model]
        transformed = self.input_scaler.transform(feature)
        normalized = np.mean(
            [model.predict(transformed) for model in models], axis=0
        )
        output = self.action_scaler.inverse_transform(normalized)[0]
        # New datasets store the IK pose before the demonstrator's PD controller
        # clips its torque.  Learning this smooth intent avoids averaging many
        # saturated, discontinuous action labels.  Old saved policies remain
        # loadable and continue to emit actions directly.
        if getattr(self, "output_kind", "action") == "joint_target":
            state = np.asarray(state, dtype=float)
            output = self.teacher_kp * (output - state[:7]) - self.teacher_kd * state[7:14]
            output = np.clip(output, -self.torque_limit, self.torque_limit)
        return output


@dataclass
class RegionalBehaviorPolicy:
    """Select one of two offline intent experts from the latched reset region."""

    standard_policy: OfflineBehaviorPolicy
    turned_policy: OfflineBehaviorPolicy

    def predict(self, state, goal_xy, direction, phase, orientation_turn=0.0):
        policy = self.turned_policy if float(orientation_turn) >= 0.5 else self.standard_policy
        return policy.predict(
            state, goal_xy, direction, phase, orientation_turn
        )


def train_behavior_policy(data: dict, cfg):
    uses_orientation_context = "orientation_turn" in data
    orientation_context = np.asarray(
        data.get("orientation_turn", np.zeros(len(data["x"]))), dtype=float
    )
    features = np.asarray(
        [
            np.r_[
                behavior_output(state, goal),
                cfg.policy_direction_weight * direction,
                np.eye(len(PHASE_NAMES))[int(phase)],
            ]
            for state, goal, direction, phase in zip(
                data["x"],
                data["goal"],
                data["direction"],
                data["phase"],
            )
        ],
        dtype=float,
    )
    if uses_orientation_context:
        features = np.column_stack([features, orientation_context])
    # Exploration-only arm warm-ups are valuable excitation for DeePC, but are
    # intentionally not examples of task-directed control.  Exclude them from
    # the supervised prior.
    variants = np.asarray(data.get("variant", np.zeros(len(features))), dtype=int)
    steps = np.asarray(data.get("step", np.zeros(len(features))), dtype=int)
    warmup = np.asarray(cfg.collection_warmup_steps, dtype=int)
    directed = steps >= warmup[np.clip(variants, 0, len(warmup) - 1)]
    output_kind = "joint_target" if "q_target" in data else "action"
    targets = (
        np.asarray(data["q_target"], dtype=float)
        if output_kind == "joint_target"
        else np.asarray(data["u"], dtype=float)
    )
    features = features[directed]
    targets = targets[directed]
    input_scaler = StandardScaler().fit(features)
    action_scaler = StandardScaler().fit(targets)
    width = int(cfg.policy_hidden_width)
    transformed_features = input_scaler.transform(features)
    transformed_targets = action_scaler.transform(targets)
    models = []
    for member in range(int(cfg.policy_ensemble_size)):
        model = MLPRegressor(
            hidden_layer_sizes=(width, width),
            activation="tanh",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=int(cfg.policy_max_iterations),
            early_stopping=True,
            n_iter_no_change=12,
            random_state=int(cfg.policy_random_seed + 997 * member),
            verbose=False,
        ).fit(transformed_features, transformed_targets)
        models.append(model)
    model = models[0] if len(models) == 1 else models
    return OfflineBehaviorPolicy(
        input_scaler=input_scaler,
        action_scaler=action_scaler,
        model=model,
        direction_weight=float(cfg.policy_direction_weight),
        output_kind=output_kind,
        teacher_kp=float(cfg.teacher_kp),
        teacher_kd=float(cfg.teacher_kd),
        torque_limit=float(cfg.torque_limit),
        uses_orientation_context=uses_orientation_context,
    )


def save_behavior_policy(policy: OfflineBehaviorPolicy, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(policy, path)
    return path


def load_behavior_policy(path: str | Path):
    return joblib.load(Path(path))
