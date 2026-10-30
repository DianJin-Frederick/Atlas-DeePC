
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np

from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.atlas import candidate_goal_routing_experiment as goals
class UnstandardizedAtlasTests(unittest.TestCase):
    def setUp(self):
        self.H = np.random.default_rng(41).normal(size=((routed.pipe.M + routed.pipe.P) * routed.pipe.L, 200))

    def test_raw_configuration_coordinates_and_online_match(self):
        config = routed.RoutedDeePCConfig()
        features, mean, scale = routed.behavioral_features(
            self.H, config.clustering_input_weight, config.clustering_feature_mode, standardize=False)
        U_p, _, Y_p, _ = routed.pipe.partition_hankel_rows(self.H)
        states = Y_p.reshape(routed.pipe.T_INI, routed.pipe.P, -1)[-1]
        np.testing.assert_array_equal(features[:, :routed.pipe.M * routed.pipe.T_INI], 0.)
        np.testing.assert_array_equal(features[:, routed.pipe.M * routed.pipe.T_INI:],
                                      states[routed.pipe.reacher_joint_configuration_rows()].T)
        np.testing.assert_array_equal(mean, 0.)
        np.testing.assert_array_equal(scale, 1.)
        for i in (0, 30, 199):
            online = routed.online_behavioral_feature(U_p[:, i:i+1], Y_p[:, i:i+1], mean, scale,
                         config.clustering_input_weight, config.clustering_feature_mode)
            np.testing.assert_array_equal(online, features[i])

    def test_standardized_default_is_unchanged(self):
        config = routed.RoutedDeePCConfig()
        default = routed.behavioral_features(self.H, 0., config.clustering_feature_mode)
        explicit = routed.behavioral_features(self.H, 0., config.clustering_feature_mode, standardize=True)
        for a, b in zip(default, explicit):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(default[0][:, 4:].std(axis=0), 1.)

    def test_atlas_uses_raw_metric_for_membership_and_graph(self):
        config = replace(routed.RoutedDeePCConfig(), clusters=3)
        atlas = routed.fit_behavioral_atlas(self.H, config, standardize=False)
        features, _, _ = routed.behavioral_features(self.H, 0., config.clustering_feature_mode, standardize=False)
        np.testing.assert_array_equal(routed.assign_charts(atlas["centers"], features), atlas["labels"])
        delta = atlas["centers"][:, None, :] - atlas["centers"][None, :, :]
        np.testing.assert_allclose(atlas["center_distance"], np.sqrt(np.mean(delta**2, axis=2)))

    def test_eight_output_projection_preserves_all_retained_samples(self):
        source = np.arange(144 * 9).reshape(144, 9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npy"
            np.save(path, source)
            reduced = experiment.load_hankel(path, output_dimension=8)
            np.testing.assert_array_equal(np.load(path), source)
        self.assertEqual(reduced.shape, (120, 9))
        np.testing.assert_array_equal(reduced[:24], source[:24])
        np.testing.assert_array_equal(reduced[24:].reshape(12, 8, 9),
                                      source[24:].reshape(12, 10, 9)[:, :8])

    def test_eight_output_features_measurement_and_dimension_scope(self):
        previous = routed.pipe.P
        with experiment.output_context(8):
            self.assertEqual(routed.pipe.P, 8)
            self.setUp()
            self.test_raw_configuration_coordinates_and_online_match()
            self.test_standardized_default_is_unchanged()
            self.test_atlas_uses_raw_metric_for_membership_and_graph()
            obs = np.arange(11, dtype=float) / 10
            expected = np.r_[obs[:4], obs[6:8], obs[8:10] + obs[4:6]]
            np.testing.assert_array_equal(routed.pipe.reacher_output_from_obs(obs), expected)
            np.testing.assert_array_equal(routed.pipe.reacher_joint_configuration_rows(), np.arange(4))
        self.assertEqual(routed.pipe.P, previous)

        with self.assertRaisesRegex(RuntimeError, "test restoration"):
            with experiment.output_context(8):
                raise RuntimeError("test restoration")
        self.assertEqual(routed.pipe.P, previous)

    def test_reference_modes_have_distinct_cache_signatures(self):
        config = experiment.default_config()
        with patch.object(experiment, "file_digest", return_value="data"), \
             patch.object(experiment, "source_hashes", return_value={}):
            for dimension in (8, 10):
                graph = experiment.execution_signature(config, "data", "atlas", dimension, "graph")
                terminal = experiment.execution_signature(config, "data", "atlas", dimension, "terminal")
                self.assertEqual(graph["reference_mode"], "graph")
                self.assertEqual(terminal["reference_mode"], "terminal")
                self.assertEqual(graph["candidate_count"], 10)
                self.assertEqual(terminal["candidate_count"], 1)
                self.assertEqual(terminal["terminal_reference_policy"], "fixed_nearest_eligible_fingertip")
                fixed_graph = experiment.execution_signature(
                    config, "data", "atlas", dimension, "graph",
                    terminal_reference_policy="fixed_nearest_eligible_fingertip")
                self.assertEqual(fixed_graph["candidate_count"], 1)
                self.assertEqual(fixed_graph["terminal_reference_policy"], terminal["terminal_reference_policy"])
                self.assertNotEqual(fixed_graph, graph)
                policy_keys = {"reference_mode", "candidate_count", "terminal_reference_policy"}
                self.assertEqual({k: v for k, v in graph.items() if k not in policy_keys},
                                 {k: v for k, v in terminal.items() if k not in policy_keys})
                with tempfile.TemporaryDirectory() as directory:
                    (Path(directory) / "manifest.json").write_text(json.dumps(graph))
                    with self.assertRaisesRegex(ValueError, "Incompatible rollout cache"):
                        experiment.load_panel([311], config, "atlas", directory, "data",
                                              output_dimension=dimension, reference_mode="terminal")
        with self.assertRaisesRegex(ValueError, "reference_mode"):
            experiment.execution_signature(config, "data", "atlas", reference_mode="invalid")

    def test_fixed_nearest_selection_is_graph_free_and_frozen(self):
        windows = routed.WINDOWS_PER_EPISODE
        for dimension in (8, 10):
            with experiment.output_context(dimension):
                H = np.zeros(((2 + dimension) * routed.pipe.L, 2 * windows))
                states = H[24:].reshape(12, dimension, -1)[1].T
                states[:, 6:8] = .5
                states[0, 6:8] = 0.0  # Closest, but excluded by the source-target separation rule.
                states[windows, 6:8] = (.01, 0.)
                states[windows+1:windows+3, 6:8] = (.001, 0.)  # Tie: lower offline index wins.
                states[:, :4] = (1., .5, 0., .8)
                states[:, 4:6] = (2., 3.)
                atlas = {"labels": np.arange(2*windows) % 3}
                with patch.object(goals, "offline_environment_targets", return_value=np.array([[0., 0.], [.2, .2]])):
                    policy = goals.FixedNearestGoalPolicy(H, atlas, experiment.default_config())
                policy.prepare(np.zeros(2))
                self.assertEqual(policy.select(0)[1]["data_goal_index"], windows+1)
                np.testing.assert_array_equal(policy.select(0)[0][:4], states[windows+1, :4])
                np.testing.assert_array_equal(policy.select(0)[0][4:8], 0.)
                self.assertIs(policy.select(0), policy.select(999))
                self.assertTrue(np.isnan(policy.select(0)[3]))
                with self.assertRaisesRegex(RuntimeError, "does not use graph"):
                    policy.path(0, 1)

    def test_fixed_controller_does_not_build_shortest_paths(self):
        baseline = experiment.baseline
        with patch.object(routed, "precompute_chart_matrix_cache", return_value={}), \
             patch.object(baseline, "shortest_path_tables", side_effect=AssertionError("Graph called")):
            controller = baseline.BaselineAtlasDeePC(
                self.H, {}, experiment.default_config(),
                terminal_reference_policy="fixed_nearest_eligible_fingertip")
        self.assertIsNone(controller.route_tables)
        with patch.object(baseline, "shortest_path_tables", return_value=("distances", "parents")) as tables, \
             patch.object(baseline, "FixedNearestGoalPolicy") as policy, \
             patch.object(routed, "run_routed_deepc", return_value={}) as rollout:
            controller.run(211, reference_mode="graph")
            controller.run(212, reference_mode="graph")
            tables.assert_called_once()
            self.assertEqual(policy.call_args.args[3], ("distances", "parents"))
            self.assertFalse(rollout.call_args.kwargs["summary_metadata"]["dynamic_terminal_reference"])
            self.assertTrue(rollout.call_args.kwargs["summary_metadata"]["graph_waypoints_enabled"])
            controller.run(213, reference_mode="terminal")
            self.assertIsNone(policy.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
