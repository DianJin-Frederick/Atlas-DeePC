"""Protect the scheduler-only boundary of the Reacher adaptation pilot."""
import unittest
import numpy as np
from atlas_deepc.benchmarks import gs_deepc_scheduler_pilot as pilot
from atlas_deepc.tests.test_gain_scheduled_deepc_reacher import GainScheduledTests


class SchedulerPilotTests(unittest.TestCase):
    def test_proxy_mapping_and_preservation(self):
        q1, q2 = np.array([-.7,.3,3.1]), np.array([.9,-1.2,.4])
        y = np.arange(24, dtype=float).reshape(3,8)
        y[:,0], y[:,1], y[:,2], y[:,3] = np.cos(q1), np.cos(q2), np.sin(q1), np.sin(q2)
        original = y.copy()
        for scheduler, expected in [('q1',q1),('q2',q2),('link2',q1+q2)]:
            proxy = pilot.proxy_observation(y, scheduler)
            np.testing.assert_allclose(np.exp(1j*pilot.gs.scheduling_variable(proxy)), np.exp(1j*expected))
            np.testing.assert_array_equal(proxy[:,[0,2,4,5,6,7]], y[:,[0,2,4,5,6,7]])
        np.testing.assert_array_equal(y, original)

    def test_q2_adapter_matches_unmodified_controller(self):
        original = GainScheduledTests().make_controller()
        adapted = pilot.ScheduledController(original.library, 'q2')
        history = np.zeros((1,8)); history[0,1] = 1.; history[0,6:8] = [.05,-.02]
        args = (np.array([[.1,-.1]]), history, np.array([.1,.05]))
        u0, row0 = original.action(*args)
        u1, row1 = adapted.action(*args)
        np.testing.assert_allclose(u0,u1,atol=1e-10)
        for key in ('planned_u','planned_y','g_coefficients','slack','objective','composite_index'):
            np.testing.assert_allclose(row0[key],row1[key],atol=1e-10)
        self.assertEqual(row0['scheduler_q2'],row1['scheduler_value'])


if __name__ == '__main__':
    unittest.main()
